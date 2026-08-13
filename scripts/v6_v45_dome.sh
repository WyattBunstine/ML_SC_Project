#!/bin/bash
# Electron-doped support experiment: bring the V6 NEMAD/ICSD expansion up to
# v4.5 (fixed BVS) and run the rung-20 LSCO-holdout dome on it. V6 roughly
# doubles electron-doped cuprate positives (69 -> 121, incl. 26 more T'-214
# Nd/Pr/Sm-Ce rows) — tests whether the electron branch failure is data support.
cd "$(dirname "$0")/.." || exit 1
STATUS=model_data/cf_calib/v6dome_status.txt
note() { echo "$(date '+%m-%d %H:%M') $*" >> "$STATUS"; }
die()  { note "HALT: $*"; exit 1; }
: > "$STATUS"
note "v6-v45 dome chain started"

note "1. rebuild NEMAD graphs (1,794) at v4.5"
rm -rf database/datafiles/MP/graphs_v5_nemad_v45 database/datafiles/MP/SC_MP_V5_nemad_v45.pickle database/datafiles/MP/SC_MP_V5_nemad_v45.csv
python main.py build-db --kind cgv4 --has-header \
  --source database/datafiles/MP/SC_MP_V5_nemad_build.csv database/datafiles/MP/cifs_v5_nemad/ \
  --output database/datafiles/MP/SC_MP_V5_nemad_v45 \
  --graph-dir database/datafiles/MP/graphs_v5_nemad_v45 \
  --oxidation-parent-csv database/datafiles/MP/SC_MP_V5_nemad_parentmap.csv \
  > model_data/cf_calib/v6_nemad_build.log 2>&1 || die "nemad build failed"
note "2. rebuild ICSD graphs (285) at v4.5"
rm -rf database/datafiles/MP/graphs_v5_icsd_v45 database/datafiles/MP/SC_MP_V5_icsd_v45.pickle database/datafiles/MP/SC_MP_V5_icsd_v45.csv
python main.py build-db --kind cgv4 --has-header \
  --source database/datafiles/MP/SC_MP_V5_icsd_build.csv database/datafiles/MP/cifs_v5_icsd/ \
  --output database/datafiles/MP/SC_MP_V5_icsd_v45 \
  --graph-dir database/datafiles/MP/graphs_v5_icsd_v45 \
  --oxidation-parent-csv database/datafiles/MP/SC_MP_V5_icsd_parentmap.csv \
  > model_data/cf_calib/v6_icsd_build.log 2>&1 || die "icsd build failed"

note "3. remap V6 index onto v45 graph dirs + pack"
python3 - <<'PYEOF' >> "$STATUS" 2>&1 || die "index remap/pack failed"
import os, sys, json, pandas as pd
sys.path.insert(0, "models/common"); sys.path.insert(0, "database")
import crystal_graph_v4_import  # noqa
from pack import pack_dataset, PackedCIFDataV4

d = pd.read_pickle("database/datafiles/MP/SC_MP_V6_expanded.pickle")
REMAP = {"graphs_v4_doped_oxifix": "graphs_v4_doped_v45",
         "graphs_v5_nemad": "graphs_v5_nemad_v45",
         "graphs_v5_icsd": "graphs_v5_icsd_v45"}
def remap(p):
    d_, b = os.path.dirname(p), os.path.basename(p)
    return os.path.join(os.path.dirname(d_), REMAP[os.path.basename(d_)], b)
d["graph_path"] = d.graph_path.map(remap)
ok = d.graph_path.map(os.path.exists)
print(f"remap: {int((~ok).sum())} of {len(d)} graphs missing after v45 rebuild (dropped)")
d = d[ok].reset_index(drop=True)
g = json.load(open(d.graph_path.iloc[-1]))
assert g.get("cf_schema") == 4 and "bvs" in g["nodes"][0], "expansion graphs not v4.5!"
d.to_pickle("database/datafiles/MP/SC_MP_V6_v45.pickle")
d.drop(columns=[c for c in ("graph",) if c in d]).to_csv("database/datafiles/MP/SC_MP_V6_v45.csv", index=False)
import shutil
shutil.rmtree("database/datafiles/MP/SC_pack_v6_v45", ignore_errors=True)
pack_dataset("database/datafiles/MP/SC_MP_V6_v45.pickle", "database/datafiles/MP/SC_pack_v6_v45")
h = json.load(open("database/datafiles/MP/SC_pack_v6_v45/pack_header.json"))
assert h.get("has_cf") and h.get("has_bvs"), h
pk = PackedCIFDataV4("database/datafiles/MP/SC_pack_v6_v45", build_angle_bias=True,
                     use_valence_features=True, use_cf_features=True, use_bvs_features=True)
assert pk[0][0][0].shape[-1] == 32
print(f"SC_pack_v6_v45 OK: {h['n_samples']:,} rows, dim 32, has_cf+has_bvs")
PYEOF

note "4. wait for GPU (hurdle run) then dome: rung 20 on V6-v45"
while pgrep -f "train-head configs/head/gps_tc_v4_la_series_20_hurdle" >/dev/null; do sleep 60; done
python3 - <<'PYEOF' || die "config write failed"
import json
c = json.load(open("configs/head/gps_tc_v6_lsco_holdout.json"))
c["name"] = "gps_tc_v6v45_lsco_20"
c["checkpoint"] = "model_data/2026-08-12/gps_mt_20_cf_bvs_fixed_nd/gps_mt_20_cf_bvs_fixed_nd_2026-08-12_22-32-02/result_cf_bvs_f_nd_model_best.pth.tar"
c["embed_source"] = "database/datafiles/MP/SC_pack_v6_v45"
c["index_path"] = "database/datafiles/MP/SC_MP_V6_v45.pickle"
c["embed_dir"] = "database/datafiles/MP/embeddings/v6v45_lsco_20"
json.dump(c, open("configs/head/gps_tc_v6v45_lsco_20.json", "w"), indent=2)
PYEOF
python main.py train-head configs/head/gps_tc_v6v45_lsco_20.json \
  > model_data/cf_calib/v6dome20.log 2>&1 || die "V6 dome FAILED (see v6dome20.log)"
note "dome: $(grep -oE '7-seed ensemble: .*' model_data/cf_calib/v6dome20.log | tail -1)"
RUN=$(ls -dt model_data/*/gps_tc_v6v45_lsco_20_2* 2>/dev/null | head -1)
note "run dir $RUN"
python scripts/plot_lsco_dome.py "$RUN/predictions.csv" \
  "docs/figures/tc_vs_cu_oxidation_la_series_dome_20_v6.png" \
  'La$_2$CuO$_4$ family doping series — rung 20 on V6 (2x electron-doped support)' \
  >> model_data/cf_calib/v6dome20.log 2>&1 && note "figure written"
python scripts/dome_stats.py "$RUN" >> "$STATUS" 2>/dev/null
note "V6-V45 DOME CHAIN COMPLETE — compare electron branch vs dome20 (Ce rows pred 0.0)"
