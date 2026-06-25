# Multi-GPU (DDP) scope for the GPS multitask pretraining

**Status: SCOPE / PLAN ONLY (2026-06-25). Not implemented.** Requested as the
high-effort speedup option after TF32 + frame-subsampling (the cheap wins, done).

## Baseline & motivation
A prior MPtrj-scale GPS-family run logged **99% GPU util, ~1% data-wait** — the
trainer is compute-bound, so a second/third/fourth GPU translates almost directly to
throughput. We currently request **1 GPU** (`SLURM_GPUS=1`); Rockfish a100 nodes have
~4. The single-process loop in `models/common/train.py` (`run_multitask`) has no
distributed code.

## ⚡ Zero-effort alternative — read this first
If the goal is **"the 4-rung ladder finishes sooner"** (not "one rung trains faster"),
you do NOT need DDP. Launch the four rungs as four independent 1-GPU jobs (concurrently
on a 4-GPU node, or as four separate submissions). Same total GPU-hours, but the ladder
completes in ~1 rung's wall-clock instead of 4×. No code change. **DDP is only worth it
when a SINGLE rung must be faster** (e.g. iterating on rung 04 alone, or scaling the
model past one GPU's memory).

## What DDP requires here (the real work)

1. **Launch + process group.** `torchrun --nproc_per_node=N models/GPSTransformer/gps_main.py`
   (or `mp.spawn`); `dist.init_process_group("nccl")`; `torch.cuda.set_device(local_rank)`.
   `deploy.sh` must set `SLURM_GPUS=4`, request matching `cpus`, and `srun torchrun ...`.
   Rank 0 owns all I/O (checkpoints, `_epoch_log.csv`, ResourceMonitor); other ranks stay silent.

2. **Model wrap.** `DistributedDataParallel(model, device_ids=[local_rank])`.
   - Every head runs every step (energy/forces/stress/magmom/bandgap/dos all computed),
     so **`find_unused_parameters=False`** is correct (no per-step unused params).
   - **Double-backward sharp edge (the #1 risk):** conservative forces do
     `autograd.grad(E, cart, create_graph=True)` *inside* forward, then `loss.backward()`.
     DDP's gradient allreduce hooks fire on the OUTER backward; higher-order graphs are a
     known DDP rough spot. Likely needs **`static_graph=True`** (the graph topology is
     fixed across steps). Must be validated, not assumed — adapt `verify_multitask_train.py`
     to a 2-rank run and assert param grads match the single-GPU reference within tolerance.

3. **Data sharding + the size-grouped-sampler desync (the #1 implementation cost).**
   DDP needs each rank to see a **disjoint** shard AND run the **same number of optimizer
   steps** — unequal step counts → NCCL allreduce hangs. Our `SizeGroupedBatchSampler`
   emits a **variable batch count** depending on each shard's atom distribution, so naive
   sharding desyncs. Plan:
   - Deterministically partition indices across ranks each epoch (shard by material so the
     material split stays leakage-free), seeded by epoch.
   - Each rank builds its size-grouped batches over its shard (reuse the existing sampler).
   - **All-reduce the per-rank batch count to the min** and have every rank iterate exactly
     that many batches (drop the tail). Small per-epoch data loss; avoids the hang.
   - `BalancedEpochSampler` (train re-sampling) composes underneath the per-rank shard.

4. **Stats consistency.** Compute `compute_feature_stats` / `compute_target_stats` on rank 0
   (or per-shard) and **broadcast from rank 0** so every replica normalizes identically.

5. **Validation.** Simplest: rank 0 validates (`_validate_mt`, val set is small) and
   broadcasts the metric for the checkpoint decision. (Sharded val + all-reduce is a later
   optimization.) Validation needs `enable_grad` for forces — already handled.

6. **LR / schedule.** Effective batch is N× larger. The warmup-steps math
   (`_resolve_warmup_steps`, per-step) changes with fewer steps/epoch; revisit warmup and
   consider a modest LR scale. Tuning item, not correctness.

## Effort & payoff
- **Effort:** ~1–2 focused days. Phase 1: single-node 4-GPU, material-sharded indices,
  rank-0 val/ckpt/log, `static_graph=True`. Phase 2: the min-batch-count sync for the
  size-grouped sampler. Phase 3: 2-rank double-backward correctness gate.
- **Payoff:** ~3–3.5× on 4 GPUs (sublinear: allreduce + rank-0 validation + tail-drop).
- **Risks:** (a) double-backward × DDP allreduce correctness — test it; (b) size-grouped
  batch-count desync hang — the min-count sync is the fix; (c) NCCL/module setup on Rockfish.

## Recommendation
Try the **zero-effort concurrent-rungs** approach first — it likely solves the ladder
wall-clock with no code. Reserve DDP for when a single rung's per-epoch time is the
blocker. If we build DDP, do Phase 1 behind a `--distributed` flag so the single-GPU
path stays byte-identical and the existing gates keep passing unchanged.
