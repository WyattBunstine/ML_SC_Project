# Multi-GPU (data-parallel) for the GPS multitask pretraining

**Status: IMPLEMENTED (2026-06-25), branch master.** Data-parallel training for the
GPS multitask pretrainer (`run_multitask`), gated so the single-GPU path is
byte-identical. Validated by `scripts/verify_ddp_multitask.py` (2 ranks, gloo).

## Why explicit all-reduce, not `DistributedDataParallel`
The conservative-force path computes `autograd.grad(E, cart, create_graph=True)` INSIDE
the forward, so `loss.backward()` is a *second* backward — exactly where DDP's
gradient-allreduce autograd hooks are most fragile. We instead keep the plain model on
each rank and synchronize explicitly (`models/common/dist_utils.py`):

- **`broadcast_model`** (startup): copy rank-0's parameters AND buffers (the feature-norm
  stats) to every rank, so all replicas start bit-identical.
- **`all_reduce_grads`** (every step, after `loss.backward()`, before clip/step): average
  the gradients across ranks via one coalesced collective, so every replica steps
  identically and stays in lockstep.

Robust for double-backward and trivially testable; cost is ~10% less compute/comm overlap
than the DDP wrapper — the right trade here. If we ever want that overlap back, the DDP
wrapper with `static_graph=True` is the drop-in alternative.

## How it works (as built)
- **Launch.** `deploy.sh run <gps multitask config>` with `slurm.gpus > 1` emits
  `torchrun --standalone --nproc_per_node=<gpus> models/GPSTransformer/gps_main.py <config>`
  under one `srun`/`ntasks=1`. `gps_main` calls `init_distributed()` (reads torchrun's
  `RANK`/`LOCAL_RANK`/`WORLD_SIZE`); NCCL when there's a GPU per rank, gloo otherwise (the
  CPU test). No torchrun → `WORLD_SIZE=1` → single-process, unchanged.
- **Data sharding.** `DistributedShardSampler` (data.py) gives each rank a disjoint,
  per-epoch-reshuffled shard of the train indices, fed UNDER `SizeGroupedBatchSampler`
  (size-grouping within the shard). The material split is computed identically on all ranks
  first, so sharding stays leakage-free. `set_epoch(epoch)` is called each epoch.
- **Equal steps.** The size-grouped sampler yields a different batch count per shard, so
  each epoch all-reduces the **min** count (`all_reduce_min_int`) and every rank runs exactly
  that many steps (drops its tail) — unequal counts would deadlock the next collective.
- **Stats.** `feature_stats` (model buffers, via `broadcast_model`) and `target_stats`
  (`broadcast_object`) are computed once on rank 0 and propagated, so all replicas normalize
  identically.
- **Rank 0 only:** validation (`_validate_mt`; replicas are identical, so one validates),
  checkpoint, epoch-log CSV, ResourceMonitor. A per-epoch `barrier` keeps ranks aligned.
- **Collective abort.** A non-finite loss (per step) or NaN val loss aborts ALL ranks
  together via an all-reduced finite-flag — a per-rank `sys.exit` would otherwise strand
  the survivors at the next collective until walltime. `all_reduce_grads` reduces every
  `requires_grad` param (not just those with a grad this step), so the reduction can't
  misalign even if a future head's gradient becomes batch-conditional.
- **Workers.** `num_workers` is divided across ranks (`num_workers // world_size`) to avoid
  CPU oversubscription; `slurm.cpus` must cover all ranks (rung configs: gpus=4, cpus=24,
  workers=20 → 5/rank → 20 procs + mains < 24).
- **LR.** Kept at the configured value (no large-batch scaling) — the conservative choice
  given the earlier lr=0.003 → collapse. Warmup is specified in *epochs*, which is invariant
  to sharding (each epoch still covers all data across ranks), so no warmup change was needed.

## Running it
```bash
./scripts/deploy.sh sync-code            # ships dist_utils + the trainer/loader/deploy changes
./scripts/deploy.sh run configs/gps_mt_ablation_suite/04_dos_full.json   # slurm.gpus=4 -> torchrun x4
```
The four MT rung configs carry `slurm.gpus=4`. Expected ~3–3.5× per-rung speedup (sublinear:
allreduce + rank-0 validation + tail-drop). Checkpoints/logs are unchanged in layout (rank 0
writes them), so `embed-gps` / `train-head` / `compare_runs` downstream are unaffected.

## Gate
`scripts/verify_ddp_multitask.py` runs 2 gloo ranks over a synthetic multitask pack and asserts
the two invariants that make the scheme correct: (1) **replica consistency** — ranks start
identical (broadcast) and STAY identical after every all-reduced step (a param-checksum's
cross-rank max==min); (2) **double-backward under DDP** — a force-only objective still yields
nonzero parameter gradients on each rank. Runs on CPU, so it gates on any box (incl. CI / the
1-GPU dev machine) BEFORE a real NCCL multi-GPU run.

## Known limitations / future
- Wired for the **multitask** GPS trainer only; single-target GPS regression under torchrun
  exits with a message (run it on one GPU).
- Single-node only (`--standalone`); multi-node would need a rendezvous endpoint.
- The masked per-task loss is normalized per-batch then grad-averaged across ranks, so a rank
  whose batch has few/no samples for a task slightly dilutes that step's gradient for it.
  Material-sharding keeps this balanced in expectation; it's a minor stochastic effect.
