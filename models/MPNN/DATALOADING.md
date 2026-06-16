# MPNN data pipeline: residence modes & the large-dataset path

How the datasets (`MPNNData.py` / `MPNNPack.py`) feed the GPU, and how to scale
past memory. Written 2026-06; updated for the packed columnar store.

## TL;DR

- The graphs are pre-computed JSON on disk (`graph_path` in the index pickle).
  Turning one into model-ready padded tensors ("**building**") costs ~4 ms/graph
  and is the dominant CPU cost. JSON read is only ~2 ms; the sort/pad/stack is the
  rest.
- A fully-built sample is small: **~38 KB** poly-off, **~181 KB** poly-on. So the
  whole MP-energy set (~49k graphs) is **~1.9 GB / ~8.9 GB** built — fits in RAM or
  an A100 many times over.
- Four **modes** exist today, selected by config. For large datasets the answer
  is the **packed columnar store** (mode 4): pack once with
  `python main.py pack-dataset`, point `index_path` at the pack directory, and
  reads drop from ~4 ms to ~0.1-0.35 ms/sample (no JSON parse, no build — just a
  memmap slice + pad). For small datasets that fit in memory, prebuild also works.

## The three stages you care about are already a DataLoader

The "CPU builds ahead while the GPU trains while data streams into device memory"
pipeline is not something we have to invent — `DataLoader` is that pipeline:

| Stage | Mechanism | Where |
|---|---|---|
| CPU generates graphs ahead | worker processes (`num_workers>0`), each builds `prefetch_factor` batches ahead | `get_*_loader` in `MPNNData.py` |
| Stage into device memory | `pin_memory=True` + `.cuda(non_blocking=True)` — overlaps the copy with GPU compute | `_to_device`, `MPNNMain.py` |
| Train on current batch | the training loop | `MPNNMain.py` |
| Keep a hot set resident, evict the rest | per-worker LRU cache of built tensors (`graph_cache_size`) | `_item_cache`, `MPNNData.py` |

Key consequence: **per-batch `non_blocking` transfer already fully overlaps with
compute**, so the GPU does not wait on data movement. Holding a large
*GPU-resident chunk* and swapping it buys ~nothing in throughput over per-batch
streaming — unless you reuse a chunk for many steps (chunk-local shuffling), which
trades away i.i.d. shuffle quality. The real ceilings at scale are **CPU build
rate** and **small-file disk I/O**, neither of which a GPU-chunk scheme fixes.

## Residence modes (implemented today)

Selected via config keys consumed in `MPNNMain.py`:

### 1. Full GPU residence — `prebuild_dataset: true`, `prebuild_device: "cuda"`
`CIFDataV4.prebuild("cuda")` builds every sample once and parks it in VRAM.
Per-batch host→device copy disappears. Forces `num_workers=0` (CUDA tensors can't
cross a worker boundary; nothing left to build). Startup is slower (~245k tiny H2D
copies for ~49k graphs) and it spends VRAM. Use only if profiling shows the
per-batch transfer is the bottleneck (rare).

### 2. Full CPU residence — `prebuild_dataset: true`, `prebuild_device: "cpu"` (default)
`prebuild("cpu")` builds every sample once into RAM. `num_workers=0`; each batch is
copied to the GPU pinned + `non_blocking` (a few MB, overlapped → sub-ms). **This is
the recommended default whenever the built dataset fits in RAM** — fastest startup,
no VRAM cost, identical training throughput to mode 1. `prebuild_device: "auto"`
picks `cuda` if it fits with headroom, else `cpu`.

> Note: the prebuild is **single-process on purpose**. A multiprocessing pool was
> measured *slower* (288 s vs ~210 s for ~49k graphs) because shipping ~2–9 GB of
> built tensors back from workers through pipes costs more than the parallelism
> saves, and it doubles peak RAM. A one-time ~3–4 min build is negligible against a
> multi-hour run. See `CIFDataV4.prebuild` docstring.

### 3. Streaming / lazy LRU — `prebuild_dataset: false`, `num_workers > 0`
The original path, still intact. `__getitem__` builds on demand and keeps the most
recently used `graph_cache_size` built samples per worker (LRU eviction). With
`persistent_workers` the cache survives epochs. Bounds memory while overlapping
CPU build (workers) with GPU compute (non_blocking transfer). Superseded for
large datasets by mode 4.

### 4. Packed columnar store — `index_path` pointing at a pack directory (PREFERRED at scale)
One-time `python main.py pack-dataset --index <pickle> --out <dir>` (or
`deploy.sh pack-mptrj` on the cluster) runs the graph-JSON parse + neighbor
extraction ONCE and stores the ragged result as flat memmap-able binary arrays +
offsets. `PackedCIFDataV4` (`MPNNPack.py`) is a drop-in for `CIFDataV4` — the
factory `load_cif_dataset` auto-detects a pack dir, and samples are
**bitwise-identical** to the lazy loader (both assemble through the same
`_extract_ragged`/`_assemble_sample` code). `max_num_nbr`/`max_num_poly_nbr`/
angle flags stay READ-time parameters, so one pack serves every config variant;
only a graph rebuild requires a re-pack. Measured: 0.16-0.35 ms/sample (~30x over
lazy), 1-4 workers feed an a100; on MP_Energy the same train epoch went 96 s →
12.8 s with GPU util 9% → 82%+. Caveat: a pack that dropped failed samples is
split-incompatible with its source pickle (different shuffle length) — the reader
warns; train and evaluate against the same backend.

## Turning on the fallback today (dataset > memory)

No new code needed for a first-cut streaming fallback — set in the config:

```jsonc
{
  "prebuild_dataset": false,   // do NOT try to hold the whole set resident
  "num_workers": 20,           // CPU workers building ahead (<= cluster cpus-1)
  "graph_cache_size": 200000,  // per-worker LRU budget, in #samples kept built
  // batch_size, etc. as usual
}
```

Tuning notes:
- `num_workers`: the CPU build is the throughput ceiling here. More workers → more
  build parallelism, up to `cpus_per_task - 1`. Each worker holds its own cache, so
  total RAM ≈ `num_workers × graph_cache_size × per-sample-bytes`.
- `graph_cache_size` is **count-based** today (number of built samples), not a byte
  budget. Estimate bytes ≈ count × (38 KB poly-off / 181 KB poly-on) × num_workers.
- The LRU + random sampler means the resident hot set is effectively random; on
  epoch k a worker has cached ~`1-(1-1/num_workers)^k` of what it has seen. Misses
  cost one rebuild (~4 ms), not a crash.

## Future work (NOT implemented) — in priority order

1. ~~**Shard packing.**~~ **DONE** — this is mode 4 (`MPNNPack.py`), implemented
   2026-06 with ragged columnar storage rather than per-graph shards.
2. **Byte-budgeted LRU + exposed `prefetch_factor`.** Make `graph_cache_size` a GB
   target (evict by bytes, not count) and surface `DataLoader(prefetch_factor=...)`
   in `get_*_loader` (currently the default of 2). Estimate: ~half a day.
3. **`dataset_residence: "auto"` selector.** One switch that estimates
   built-dataset-vs-available-memory and picks full-GPU → full-CPU → streaming-LRU,
   unifying modes 1–3. Estimate: ~half a day. Pairs naturally with #2.
4. **Chunked GPU double-buffer (lowest value, largest effort).** Background builder
   → host chunk → VRAM chunk with chunk-local shuffling, double-buffered. Only worth
   it combined with #1 and only if profiling shows the pipeline is transfer-bound
   (unlikely given per-batch `non_blocking` already overlaps). Changes shuffle
   semantics. Estimate: week+, rewrites the sampler/loader.

Recommendation: ship #1 + #3 (+ #2) when the need arrives; skip #4 unless a profile
demands it.
