# T_c transfer-head configs

The transfer step trains a small T_c head on a **frozen** encoder's per-structure
embeddings + the physical-descriptor bypass (`models/head/`). The encoder is never
fine-tuned — only the head (~7.5k params at `pca_k=64`, `hidden=64`) is trainable, per
the parameter budget (~5.8k noisy T_c labels → cap fresh params ~10k).

## One-command reproduction
```bash
python main.py train-head configs/head/gps_tc_3dsc_doped.json
```
With a GPS `checkpoint` set, `train-head` auto-builds anything missing, then trains:
1. **embeddings** — `embed-gps` over `embed_source` (a pack, fast) → `embed_dir/<id>.npy`;
2. **descriptors** — `build_descriptor_table` over `index_path` → `descriptors`;
3. **head** — pool embeddings ∥ descriptors → PCA-whiten → MLP → T_c.

All three are cached/resumable: an `embed_dir` with `.npy` or an existing `descriptors`
pickle is reused untouched, so a re-run only retrains the head. (The MACE configs have no
`checkpoint` and instead point `embed_dir` at a prebuilt embedding dir — those artifacts
must pre-exist.)

## The doping A/B
`gps_tc_3dsc_doped` vs `gps_tc_3dsc_baseline` differ ONLY in the SC graph set the encoder
sees — `graphs_v4_doped` (doping-aware charge-balanced oxidation + the radius/IE/EA that
follow) vs `graphs_v4_baseline` (oxidation ≈ 0, the pre-fix behavior). Same frozen
encoder, same split seed, same head hyperparameters. The hypothesis: doping-aware features
help most for the doping-tuned families (cuprates). Compare the family-resolved test MAE
from the two run dirs.

> Set `checkpoint` to the **final** pretrained encoder (`<run>/result_model_best.pth.tar`)
> before the real A/B — the committed value points at an earlier run for dry-run wiring.

## Fields
| field | meaning |
|---|---|
| `checkpoint` | frozen GPS encoder `.pth.tar` (absent → MACE-style prebuilt `embed_dir`) |
| `embed_source` | what `embed-gps` reads (a pack dir = fast, or the index pickle); defaults to `index_path` |
| `index_path` | SC index **pickle** (id/tc/label + graph_path) — used by the head and descriptors |
| `embed_dir` / `descriptors` | output/cache paths for the two artifacts |
| `metadata_csv` | 3DSC master CSV for family + `synth_doped` tags |
| `pca_k`, `hidden`, `dropout`, `n_seeds` | head capacity / ensemble |
| `class_pretrain` | SC/non-SC trunk pretrain — needs a non-SC-bearing index + their embeddings (off here; the SC-only index has no negatives) |
