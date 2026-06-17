# gps_ablation_suite — building up the GPS model one component at a time

A one-factor-per-rung ladder that grows `GPSCrystalNet` from raw atom features to the
full model, to motivate (and measure) what each component buys on MPtrj formation
energy. Every rung shares an **identical backbone** — same dataset (`packed_v1`),
`split_seed`, split, batching, and the stable optimizer settings (LR 3e-4, 3-epoch
warmup, grad-clip 0.5) — so `compare_runs.py` can do paired significance on the shared
test set. Each rung flips exactly one architectural flag relative to the previous.

| rung | adds | flags changed | question |
|---|---|---|---|
| 01_raw_mlp | nothing (baseline) | `n_conv:0` | How far does a per-atom MLP on the raw 14-dim features alone get? |
| 02_bond_mean | bond edges, no attention | `n_conv:4, use_bond_edges:true, shell_aggregation:mean` | Does aggregating bond-edge features (mean) over neighbors help at all? |
| 03_poly_mean | polyhedral edges | `use_poly_edges:true` | Does the second (polyhedral) relation add signal? |
| 04_attention | shell attention | `shell_aggregation:attention` | Attention over the shell vs. a plain mean — does the learned aggregation win? |
| 05_angle_bias | 3-body angle bias | `use_angle_bias:true` | Does the benchmark-winning angle bias replicate inside GPS? |
| 06_local_transf | local transformer FFN | `local_transformer:true` | Does making the local channel a full pre-LN transformer (attn+FFN) help? |
| 07_global_attn | within-crystal global attention | `gps_global:true` | Does crystal-global attention help energy? (expected ~neutral; its real test is T_c) |
| 08_dist_bias | PBC long-range distance bias | `use_dist_bias:true`, `index_path:packed_v2` | Does long-range geometry on the global attention help? (a T_c bet; ~neutral on energy expected) |

Rung 07 is the current full GPS encoder (== `configs/gps/gps_eform.json` minus the
distance bias). **Rung 08 requires `packed_v2`** (the positioned pack, built via
`deploy.sh augment-positions` + a re-pack) — it's the only rung that needs the data
regeneration. `packed_v1` and `packed_v2` share the same frames + `split_seed`, so
`compare_runs` still pairs 07 vs 08 on an identical test set (the only difference is
the distance bias; 07 simply ignores the positions v2 carries).

Notes / caveats:
- The 01→02 step changes two things (adds the network *and* bond edges); every later
  step is a clean single-flag delta.
- `n_conv=4` for rungs 02–07 isolates *components*, not depth — depth/width are a
  separate capacity axis, deliberately held fixed here.
- Energy is nearsighted, so rung 07's global channel is expected to be roughly neutral
  on this target; don't read its (lack of) energy gain as a verdict on T_c.
- 60 epochs / `[30,50]` milestones keep the 7-run suite affordable; bump for a final
  number. Simpler rungs (no global) train much faster than 06–07.

Launch all (each submits a separate SLURM job):
```
for f in configs/gps_ablation_suite/0*.json; do ./scripts/deploy.sh run "$f"; done
```

Compare after `./scripts/deploy.sh fetch`:
```
python scripts/compare_runs.py model_data/<date>/gps_abl_*/gps_abl_* --baseline 0 --boot 3000
```
`compare_runs.py` reads each run's `metadata.architecture` (which now records
`use_bond_edges`, `shell_aggregation`, `use_angle_bias`, `gps_global`, …) as the rung
descriptor and bootstraps the paired test-error difference vs. the baseline rung.
