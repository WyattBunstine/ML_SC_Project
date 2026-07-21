"""Shared frozen-encoder embedding cache + segment-forward helpers.

One implementation of the cache-build / cat / repeat_interleave idiom used by the
fine-tune protocol (FineTune.run phase A), the stage-A head HPO sweep
(scripts/head_hpo_sweep.py), and the encoder-zoo probe (scripts/probe_encoders.py).
Previously three near-verbatim copies — a fix to the batch layout (the inp[6]
segment slot below) had to be replicated by hand and a missed copy silently
misaligned per-atom embeddings.
"""
import torch


@torch.no_grad()
def build_embed_cache(model, loader, device):
    """Per-structure per-atom embeddings {cif_id: (n_atoms, D)} from a frozen
    encoder. inp[6] is the batch segment-id tensor emitted by collate_pool_geom —
    if the collate tuple layout ever changes, THIS is the single place to fix."""
    model.eval()
    cache = {}
    for inp, _t, _l, cif_ids in loader:
        inp = tuple(x.to(device) if torch.is_tensor(x) else x for x in inp)
        seg = inp[6].to(device).long()
        h = model.encode(*inp)
        for c, cid in enumerate(cif_ids):
            cache[cid] = h[seg == c]
    return cache


def cat_cached(cache, bids, device):
    """Concatenate cached per-atom embeddings for a batch -> (h, seg) ready for a
    TcHead forward (seg maps each atom row to its structure index in bids)."""
    h = torch.cat([cache[c] for c in bids])
    counts = torch.tensor([cache[c].shape[0] for c in bids], device=device)
    seg = torch.repeat_interleave(torch.arange(len(bids), device=device), counts)
    return h, seg
