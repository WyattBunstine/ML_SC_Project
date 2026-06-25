#!/usr/bin/env python3
"""Correctness gate for the explicit-allreduce data-parallel multitask trainer.

Runs TWO ranks under the gloo backend (CPU — so it validates on any box, incl. the
1-GPU dev machine and CI) over a shared synthetic multitask pack, exercising the
distributed primitives in common/dist_utils + the sharded loader + the double-backward
step. It asserts the two invariants that make the manual-allreduce scheme correct:

  1. REPLICA CONSISTENCY: after broadcast_model the ranks start bit-identical, and after
     every all-reduced optimizer step they STAY identical (a param checksum's cross-rank
     max == min). If the gradient sync were wrong, the replicas would drift -> caught here.
  2. DOUBLE-BACKWARD UNDER DDP: a force-only objective still produces nonzero parameter
     gradients on each rank (the create_graph=True inner grad survives the multi-process
     path), so forces actually train under data parallelism.

    python scripts/verify_ddp_multitask.py     # exit 0 = pass, 1 = fail

This is the gate to pass BEFORE a real multi-GPU (NCCL) cluster run — it catches the
hang/divergence/zeroed-force-grad failure modes without needing >1 GPU.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("models/common", "models/GPSTransformer"):
    sys.path.insert(0, os.path.join(ROOT, _p))

import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "verify_multitask_train", os.path.join(ROOT, "scripts", "verify_multitask_train.py"))
_vmt = importlib.util.module_from_spec(_spec)

WORLD = 2
STEPS = 6
CONSISTENCY_TOL = 1e-6


def _build_pack(tmp):
    """A 24-graph synthetic multitask pack (reuses verify_multitask_train's builders so
    the two gates can't drift), with mp_id so the union/material split is well-defined."""
    import pandas as pd
    from pack import pack_dataset
    _spec.loader.exec_module(_vmt)
    gd = os.path.join(tmp, "g"); os.makedirs(gd)
    ids, paths = [], []
    for i in range(24):
        p = os.path.join(gd, f"{i}.json")
        json.dump(_vmt._graph(i), open(p, "w"))
        ids.append(str(i)); paths.append(p)
    idx = pd.DataFrame({"id": ids, "graph_path": paths, "label": [1] * 24,
                        "mp_id": [f"mp-{i}" for i in range(24)],
                        "formation_energy_per_atom": [-1.0 - 0.05 * i for i in range(24)],
                        "bandgap": [1.0 + 0.1 * i for i in range(24)]})
    ip = os.path.join(tmp, "idx.pickle"); idx.to_pickle(ip)
    pk = os.path.join(tmp, "pack"); pack_dataset(ip, pk, n_workers=1)
    return pk


def _param_checksum(model):
    return sum(float(p.detach().double().abs().sum()) for p in model.parameters())


def _ranks_agree(value, di):
    """True iff `value` is identical across ranks (max == min within tol)."""
    import torch.distributed as dist
    dev = "cpu"
    hi = torch.tensor([value], dtype=torch.float64, device=dev)
    lo = hi.clone()
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    return float(hi.item() - lo.item()) < CONSISTENCY_TOL


def _worker(rank, pack_dir, result_path):
    warnings.simplefilter("ignore")
    _spec.loader.exec_module(_vmt)              # spawned child: populate _graph / TASKS
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29577",
                      RANK=str(rank), WORLD_SIZE=str(WORLD), LOCAL_RANK=str(rank))
    from data import (load_cif_dataset, get_sc_nonsc_loaders, collate_pool_multitask,
                      compute_feature_stats, resolve_split_by, DOS_N_ENERGY)
    from model import GPSCrystalNet
    from train import (compute_target_stats, _train_mt, _DEFAULT_LOSS_WEIGHTS)
    from dist_utils import (init_distributed, cleanup, broadcast_model,
                            broadcast_object, all_reduce_min_int)

    di = init_distributed()
    ok = True
    try:
        ds = load_cif_dataset(pack_dir, target_column="formation_energy_per_atom",
                              use_poly_edges=True, build_angle_bias=True, multitask=True)
        split_by = resolve_split_by("material", ds)
        # size_grouped + a tiny atom cap so batches are size-capped and the per-rank batch
        # COUNT varies — exercising the min-step sync AND the early-break _pending reset
        # (the size-grouped sampler's post-pass auto-clear never runs when we drop the tail).
        loaders = get_sc_nonsc_loaders(
            ds, batch_size=4, val_ratio=0.1, test_ratio=0.1, seed=123,
            split_by=split_by, collate_fn=collate_pool_multitask, dist_info=di,
            size_grouped=True, max_atoms_per_batch=12, size_pool_factor=2)
        sc_idx = loaders["train_sc_idx"]
        sa, sn, _, sp, _, _ = ds[0][0][:6]

        # DIFFERENT init per rank on purpose -> broadcast_model must make them identical.
        torch.manual_seed(rank + 1)
        model = GPSCrystalNet(sa.shape[-1], sn.shape[-1], poly_fea_len=sp.shape[-1],
                              atom_fea_len=32, n_conv=2, h_fea_len=32, n_h=2, n_heads=2,
                              use_poly_edges=True, gps_global=False, use_angle_bias=True,
                              tasks=_vmt.TASKS, differentiable_geometry=True,
                              n_energy=DOS_N_ENERGY)
        if di.is_main:                              # stats on rank 0, propagated by broadcast
            fs = compute_feature_stats(ds, list(sc_idx), max_graphs=24)
            model.set_feature_stats(fs["node"], fs["edge"], fs["poly"])
        broadcast_model(model, di, src=0)
        assert _ranks_agree(_param_checksum(model), di), "ranks diverged after broadcast_model"

        ts = compute_target_stats(ds, list(sc_idx), max_samples=24) if di.is_main else None
        ts = broadcast_object(ts, di, src=0)

        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        args = {"cuda": False, "learning_rate": 0.01, "print_split": 10**9,
                "grad_clip": 0.5, "warmup_epochs": 0}
        bsamp = getattr(loaders["train"], "batch_sampler", None)
        for ep in range(STEPS):
            # Mirror run_multitask's epoch prologue: set_epoch reshuffles the shard AND
            # drops the tail-truncated size-grouped layout; then sync the step count.
            bsamp.set_epoch(ep)
            max_steps = all_reduce_min_int(len(loaders["train"]), di)
            _train_mt(loaders["train"], model, opt, ep, ts, _DEFAULT_LOSS_WEIGHTS, args,
                      dist_info=di, max_steps=max_steps)
            # The decisive invariant: all-reduced steps keep every replica identical.
            if not _ranks_agree(_param_checksum(model), di):
                ok = False
                break

        # Double backward still trains the force head under the multi-process path.
        iv_batch = next(iter(loaders["train"]))[0]
        from train import _to_input_var, _build_cart_strain
        iv = _to_input_var(iv_batch, False)
        cart, strain = _build_cart_strain(iv)
        out = model(*iv, cart=cart, strain=strain)
        model.zero_grad(set_to_none=True)
        out["forces"].abs().sum().backward()
        f_grad = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        ok = ok and f_grad > 0
    except Exception as exc:  # noqa: BLE001
        print(f"[rank {rank}] EXCEPTION: {type(exc).__name__}: {exc}")
        ok = False
    finally:
        if di.is_main:
            with open(result_path, "w") as f:
                json.dump({"ok": bool(ok), "force_grad": locals().get("f_grad", 0.0)}, f)
        cleanup(di)


def main():
    warnings.simplefilter("ignore")
    if not torch.distributed.is_gloo_available():
        print("verify_ddp_multitask: SKIP (no gloo backend)."); return 0
    tmp = tempfile.mkdtemp(prefix="ddp_mt_")
    try:
        pack = _build_pack(tmp)
        res = os.path.join(tmp, "result.json")
        mp.spawn(_worker, args=(pack, res), nprocs=WORLD, join=True)
        out = json.load(open(res)) if os.path.exists(res) else {"ok": False}
        print(f"  replica-consistency across {WORLD} ranks over {STEPS} steps: "
              f"{'held' if out['ok'] else 'BROKEN'}")
        print(f"  force double-backward param-grad under DDP: {out.get('force_grad', 0):.2e}")
        print("verify_ddp_multitask: " + ("PASS" if out["ok"] else "FAIL"))
        return 0 if out["ok"] else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
