"""GPSTransformer trainer entry (regression).

GPS's own thin trainer: it builds the GPS model and drives the SHARED,
model-agnostic pieces in models/common — the data layer (loaders, feature stats,
splits) and the regression loop (common/train.py). It imports nothing from the
MPNN package, so the two model subfolders stay independent.

Usage: python models/GPSTransformer/gps_main.py <config.json>
       (or `python main.py train-gps <config.json>`)
"""

import datetime
import json
import os
import shutil
import sys
import warnings

# Reduce CUDA allocator fragmentation from the GPS poly shell-attention's large
# (N, heads, M, M) tensors. Must be set before torch initializes the CUDA allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

# Shared infra (models/common) on path, then GPS model from this dir.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from data import (load_cif_dataset, get_sc_nonsc_loaders,  # noqa: E402
                  compute_feature_stats, resolve_split_by, collate_pool_geom,
                  collate_pool_multitask, DOS_N_ENERGY, ConcatMTDataset)
from train import (Normalizer, run_regression,  # noqa: E402
                   compute_target_stats, run_multitask, _DEFAULT_LOSS_WEIGHTS)
from model import GPSCrystalNet  # noqa: E402


def _build_optimizer(model, args):
    name = args.get("optim", "AdamW")
    lr, wd = args["learning_rate"], args.get("weight_decay", 0)
    if name == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=args.get("momentum", 0.9),
                         weight_decay=wd)
    raise ValueError(f"Unsupported optim: {name}")


def main():
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        warnings.warn("Usage: gps_main.py <config.json>")
        return -1
    with open(sys.argv[1]) as f:
        args = json.load(f)
    if args.get("task", "regression") != "regression":
        sys.exit("GPSTransformer currently supports task='regression' only.")

    run_tag = args.get("run_tag", "GPS")
    now = datetime.datetime.now()
    run_id = f"{run_tag}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = os.path.join(args.get("model_data_dir", "model_data"),
                           now.strftime("%Y-%m-%d"), run_tag, run_id)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(sys.argv[1], os.path.join(run_dir, "config.json"))
    args["out_file"] = os.path.join(run_dir, os.path.basename(args.get("out_file", "result")) or "result")
    args["run_id"] = run_id
    print(f"Run output dir: {run_dir}")

    # Multitask pretraining mode: config carries a non-empty `tasks` list (e.g.
    # ["energy","forces","stress","magmom","bandgap"]). Conservative-autograd forces
    # need amp off and the per-target collate. Single-target GPS path is unchanged.
    multitask = bool(args.get("tasks"))
    args["differentiable_geometry"] = multitask   # recorded in metadata
    if multitask and "dos" in args["tasks"] and args.get("n_energy", 256) != DOS_N_ENERGY:
        sys.exit(f"n_energy ({args.get('n_energy', 256)}) must equal DOS_N_ENERGY "
                 f"({DOS_N_ENERGY}) when the 'dos' task is active: the dos head width must "
                 "match the per-structure DOS target grid.")

    # The GPS local channel always uses the bond-angle bias -> build it. index_path may be
    # a LIST for a multitask masked-union (e.g. packed_v4 + the DOS pack): each pack supplies
    # the targets it has, the rest NaN-masked. Packs MUST share the feature flags below.
    _ds_kw = dict(
        max_num_nbr=args.get("max_num_nbr", 14),
        max_num_poly_nbr=args.get("max_num_poly_nbr", 16),
        graph_cache_size=args.get("graph_cache_size", 4096),
        target_column=args.get("target_column"),
        use_bond_angles=args.get("use_bond_angles", False),
        use_poly_edges=args.get("use_poly_edges", True),
        build_angle_bias=True,
        # Physically-motivated element features (mass/group/row/#unpaired) concatenated
        # to the node vector at load time — no re-pack needed.
        use_rich_node_features=args.get("use_rich_node_features", False),
        # Per-atom 4-body torsion summary concatenated to the node vector. Needs a pack
        # re-built with dihedrals (has_dihedrals=true, e.g. packed_v3); zeros otherwise.
        use_dihedrals=args.get("use_dihedrals", False),
        # Multitask: __getitem__ yields (input, targets_dict, masks_dict, cif_id).
        multitask=multitask,
    )
    index_paths = (args["index_path"] if isinstance(args["index_path"], list)
                   else [args["index_path"]])
    if len(index_paths) > 1 and not multitask:
        sys.exit("multiple index_path entries (a masked-union) require a multitask `tasks` config.")
    _subsets = [load_cif_dataset(ip, **_ds_kw) for ip in index_paths]
    dataset = _subsets[0] if len(_subsets) == 1 else ConcatMTDataset(_subsets)

    split_by = resolve_split_by(args.get("split_by"), dataset)
    args["split_by"] = split_by
    loaders = get_sc_nonsc_loaders(
        dataset, batch_size=args["batch_size"],
        val_ratio=args["val_ratio"], test_ratio=args["test_ratio"],
        sc_to_nonsc_ratio=float("inf"),
        num_workers=args.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        seed=args.get("split_seed", 123), split_by=split_by,
        # Bound the within-crystal global-attention memory (B x Lmax^2) by batching
        # similar-sized cells; max_atoms_per_batch caps the peak (large-cell batches
        # get fewer crystals). Off by default -> unchanged behavior when unset.
        size_grouped=args.get("size_grouped_batches", False),
        max_atoms_per_batch=args.get("max_atoms_per_batch"),
        size_pool_factor=args.get("size_pool_factor", 20),
        prefetch_factor=args.get("prefetch_factor"),
        # GPS batches carry frac_coords + lattice (zeros on a positionless pack) so
        # the optional long-range distance bias has geometry. Multitask also batches
        # the per-target/-mask dicts + nbr_jimage for the autograd-force geometry.
        collate_fn=(collate_pool_multitask if multitask else collate_pool_geom))
    print(f"split_by={split_by} -> split sizes:", loaders["split_sizes"])

    sc_idx = loaders["train_sc_idx"]

    # [:6] tolerates the sample input tuple carrying trailing geometry (frac_coords,
    # lattice) for the GPS distance bias — we only need the feature dims here.
    sa, sn, _, sp, _, _ = dataset[0][0][:6]
    orig_atom_fea_len, nbr_fea_len, poly_fea_len = sa.shape[-1], sn.shape[-1], sp.shape[-1]

    if args.get("model_seed") is not None:
        torch.manual_seed(int(args["model_seed"]))

    # Single source of the architecture spec (shared with embed_gps's transfer
    # rebuild via GPSCrystalNet.from_args) so the two construction sites can't drift.
    # tasks -> multitask per-atom heads + conservative-autograd forces/stress; None
    # -> single-scalar readout (unchanged). use_dist_bias needs a positioned pack.
    model = GPSCrystalNet.from_args(
        args, (orig_atom_fea_len, nbr_fea_len, poly_fea_len))

    if args.get("normalize_features", True):
        stats = compute_feature_stats(dataset, list(sc_idx),
                                      max_graphs=args.get("feature_stat_graphs", 4000),
                                      seed=args.get("split_seed", 123))
        model.set_feature_stats(stats["node"], stats["edge"],
                                stats["poly"] if model.use_poly_edges else None)
        print("Installed per-feature input normalization")

    total = sum(p.numel() for p in model.parameters())
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump({
            "run_id": run_id, "model": "GPSCrystalNet",
            "timestamp": now.isoformat(timespec="seconds"),
            "feature_dims": {"node": orig_atom_fea_len, "edge": nbr_fea_len, "poly": poly_fea_len},
            "architecture": {k: args.get(k) for k in (
                "atom_feat_len", "n_conv", "h_feat_len", "n_hidden", "set_transformer_heads",
                "gps_global", "gps_global_heads", "gps_ffn_mult", "local_transformer",
                "per_atom_head", "use_bond_edges", "shell_aggregation", "use_angle_bias",
                "use_dist_bias", "use_rich_node_features", "use_dihedrals",
                "tasks", "differentiable_geometry", "n_energy",
                "atom_pooling", "use_poly_edges")},
            "training": {k: args.get(k) for k in (
                "optim", "learning_rate", "weight_decay", "lr_milestones",
                "warmup_epochs", "grad_clip", "epochs", "batch_size", "target_transform",
                "size_grouped_batches", "max_atoms_per_batch", "size_pool_factor", "amp")},
            "dataset": {"index_path": args.get("index_path"),
                        "target_column": args.get("target_column"),
                        "total_indexed": len(dataset), "split_by": split_by,
                        "split_sizes": loaders["split_sizes"]},
            "model_size": {"total_params": total},
        }, f, indent=2)
    print(f"Model: {total:,} params")

    args["cuda"] = torch.cuda.is_available()
    if args["cuda"]:
        model.cuda()

    optimizer = _build_optimizer(model, args)
    scheduler = MultiStepLR(optimizer, milestones=args.get("lr_milestones", [100]), gamma=0.1)
    if multitask:
        if args.get("amp"):
            warnings.warn("amp (bf16) degrades autograd force gradients; "
                          "multitask should run with amp=false.", RuntimeWarning)
        if args.get("gps_global"):
            warnings.warn("gps_global=True with multitask: the double backward retains the "
                          "(B, Lmax, Lmax) global-attention graph and can OOM. Prefer "
                          "gps_global=False (local encoder), or size_grouped_batches with a "
                          "small max_atoms_per_batch.", RuntimeWarning)
        # Conservative forces need REAL geometry: a positionless pack (all-zero lattice,
        # e.g. packed_v1) would silently train on degenerate zero bond vectors. Fail fast
        # before a multi-hour run (mirrors the distance-bias zero-lattice guard).
        if float(dataset[sc_idx[0]][0][7].abs().sum()) == 0.0:
            sys.exit("multitask training requires a POSITIONED pack (real frac_coords / "
                     "lattice); this pack's lattice is all-zero (e.g. packed_v1). Re-pack "
                     "with positions + forces (packed_v4).")
        target_stats = compute_target_stats(
            dataset, list(sc_idx), max_samples=args.get("target_stat_samples", 2000),
            seed=args.get("split_seed", 123))
        weights = {**_DEFAULT_LOSS_WEIGHTS, **args.get("loss_weights", {})}
        run_multitask(args, model, optimizer, scheduler, loaders, target_stats, weights)
    else:
        train_targets = torch.tensor([float(dataset.data[i][1]) for i in sc_idx],
                                     dtype=torch.float32)
        normalizer = Normalizer(train_targets, transform=args.get("target_transform", "none"))
        criterion = nn.L1Loss()
        run_regression(args, model, criterion, optimizer, scheduler, loaders, normalizer)


if __name__ == "__main__":
    main()
