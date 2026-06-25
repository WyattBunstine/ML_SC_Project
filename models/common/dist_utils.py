"""Minimal data-parallel (DDP-style) primitives for the multitask trainer.

We deliberately do NOT use torch's ``DistributedDataParallel`` wrapper. The
conservative-force path computes ``autograd.grad(E, cart, create_graph=True)``
INSIDE the forward, so ``loss.backward()`` is a *second* backward — exactly the
higher-order-graph case where DDP's gradient-allreduce autograd hooks are most
fragile. Instead we keep the plain model on each rank and synchronize EXPLICITLY:
broadcast the initial weights from rank 0 (so every replica starts identical), then
after each ``loss.backward()`` all-reduce the gradients to their mean (so every
replica steps identically and stays in lockstep). Robust and trivially testable
(scripts/verify_ddp_multitask.py), at the cost of ~10% less compute/comm overlap
than the wrapper — the right trade for a double-backward objective.

Everything here is a NO-OP when not launched under torchrun (``WORLD_SIZE<=1``), so
the single-GPU / single-process path is byte-identical and the non-distributed code
never pays a distributed cost.
"""
import os

import torch
import torch.distributed as dist


class DistInfo:
    """Process-group facts. ``enabled`` is False for the single-process path, where
    rank=0/world_size=1/is_main=True so every ``is_main`` guard degrades to 'always'."""

    def __init__(self, enabled, rank, local_rank, world_size, gpu_per_rank=True):
        self.enabled = enabled
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        # True when there's a dedicated GPU per rank (NCCL) — i.e. it's safe to pin this
        # rank to cuda:local_rank. False on the gloo fallback (more ranks than GPUs, the
        # CPU correctness test), where binding to local_rank would be an invalid ordinal.
        self.gpu_per_rank = gpu_per_rank

    def __repr__(self):
        return (f"DistInfo(enabled={self.enabled}, rank={self.rank}/"
                f"{self.world_size}, local_rank={self.local_rank})")


def init_distributed():
    """Detect a torchrun launch and initialize the process group. Returns DistInfo.

    No torchrun (WORLD_SIZE unset or 1) -> enabled=False (single-process path). On
    CUDA we use NCCL and pin this rank to its local GPU; CPU (the gloo correctness
    test) uses gloo. ``RANK``/``LOCAL_RANK``/``WORLD_SIZE`` are set by torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistInfo(False, 0, 0, 1)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # NCCL only when there's a GPU per rank (the real cluster run: torchrun --nproc_per_node
    # <= #GPUs). Otherwise gloo on CPU — the correctness test runs 2 ranks on a 1-GPU box,
    # where binding rank 1 to a nonexistent GPU ordinal would crash.
    use_nccl = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    dist.init_process_group(backend="nccl" if use_nccl else "gloo")
    if use_nccl:
        torch.cuda.set_device(local_rank)
    return DistInfo(True, rank, local_rank, world_size, gpu_per_rank=use_nccl)


def cleanup(di):
    if di.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier(di):
    if di.enabled:
        dist.barrier()


def broadcast_model(model, di, src=0):
    """Copy rank-``src``'s full state (parameters AND buffers — the latter carry the
    feature-normalization stats) into every other replica, so all ranks start from
    identical weights. One broadcast per tensor; startup-only, so the cost is moot."""
    if not di.enabled:
        return
    for tensor in model.state_dict().values():
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src)


def all_reduce_grads(model, di):
    """Average the gradients across ranks IN PLACE (the explicit stand-in for DDP's
    gradient reduction). Call after ``loss.backward()`` and BEFORE grad-clip/step, so
    clipping and the optimizer act on the synchronized mean gradient and every replica
    stays bit-aligned. Coalesces all grads into one tensor -> a single collective.

    Reduces over EVERY ``requires_grad`` parameter (materializing a zero for any whose
    grad is None this step), NOT just the params that happen to have a grad. The model
    structure is identical across ranks, so this guarantees the flattened tensor has the
    same shape on every rank regardless of which heads a rank's batch exercised —
    immune to a future change that makes a head's gradient conditional on batch contents
    (which would otherwise give ranks different-length grad lists and a shape-mismatched
    all-reduce)."""
    if not di.enabled:
        return
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= di.world_size
    for p, synced in zip(params, torch._utils._unflatten_dense_tensors(flat, grads)):
        if p.grad is None:
            p.grad = synced.clone()          # other ranks contributed a grad this step
        else:
            p.grad.copy_(synced)


def all_reduce_min_int(value, di):
    """Min of an int across ranks (single value, single collective). Used to align the
    per-epoch optimizer-step count: the size-grouped sampler yields a DIFFERENT batch
    count per shard, and unequal step counts deadlock the next collective, so every
    rank runs exactly ``min`` steps and drops its tail."""
    if not di.enabled:
        return value
    # Tensor device must match the BACKEND, not merely CUDA availability: NCCL needs a
    # CUDA tensor; gloo (the CPU correctness test, even on a GPU box) needs a CPU one.
    device = "cuda" if dist.get_backend() == "nccl" else "cpu"
    t = torch.tensor([int(value)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


def broadcast_object(obj, di, src=0):
    """Broadcast a picklable Python object from rank ``src`` (e.g. the target-stat std
    dict, computed once on rank 0) so every replica normalizes the loss identically."""
    if not di.enabled:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]
