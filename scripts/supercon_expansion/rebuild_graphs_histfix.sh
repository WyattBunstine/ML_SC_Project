#!/bin/bash
# Rebuild every SC graph set built after the 2026-08-24 builder rename (hist +
# shared_count were silently zero), regenerate descriptors, re-assemble the
# descriptor tables for V8..V11c (indices reference unchanged graph paths), sync.
cd "$(dirname "$0")/../.." || exit 1
S=model_data/cf_calib/histfix_sc_status.txt; L=model_data/cf_calib/histfix_sc.out
note(){ echo "$(date '+%m-%d %H:%M') $*" >> $S; }
note "histfix SC chain started (graph dirs deleted first: the CIF-path builder is resumable and skips existing JSONs)"
for b in build_supercon_v7 build_cuprate_negatives build_feox_negatives; do
  python scripts/supercon_expansion/$b.py graphs >> $L 2>&1; tail -40 $L | grep -q "graphs DONE" || { note "HALT $b graphs"; exit 1; }
  note "$b: $(grep 'graphs DONE' $L | tail -1 | cut -c1-70)"
done
python - >> $L 2>&1 <<'PY' || { note "HALT descriptors"; exit 1; }
import sys, pandas as pd; sys.path.insert(0,"models/head")
from descriptors import build_descriptor_table
MP="database/datafiles/MP"
for idx,out in (("SC_MP_V7_supercon","descriptors_v7_supercon"),("SC_MP_V9_cuneg","descriptors_v9_cuneg_only"),("SC_MP_V11_feox","descriptors_v11_feox_only")):
    build_descriptor_table(f"{MP}/{idx}.pickle", f"{MP}/{out}.pickle")
d45=pd.read_pickle(f"{MP}/descriptors_doped.pickle"); dn=pd.read_pickle(f"{MP}/descriptors_v4_plus_nickelates.pickle")
d7=pd.read_pickle(f"{MP}/descriptors_v7_supercon.pickle"); d9=pd.read_pickle(f"{MP}/descriptors_v9_cuneg_only.pickle"); d11=pd.read_pickle(f"{MP}/descriptors_v11_feox_only.pickle")
base=dict(d45["table"]); base.update(dn["table"]); base.update(d7["table"])
for name, extra in (("descriptors_v8_supercon",{}),("descriptors_v9_cuneg",d9["table"]),("descriptors_v10",d9["table"]),("descriptors_v11",{**d9["table"],**d11["table"]}),("descriptors_v11c",{**d9["table"],**d11["table"]})):
    idx=pd.read_pickle(f"{MP}/{ {'descriptors_v8_supercon':'SC_MP_V8_supercon','descriptors_v9_cuneg':'SC_MP_V9_cuneg','descriptors_v10':'SC_MP_V10','descriptors_v11':'SC_MP_V11','descriptors_v11c':'SC_MP_V11c'}[name] }.pickle")
    t={**base, **extra}; keep=set(idx.id); tab={k:v for k,v in t.items() if k in keep}; miss=len(keep-set(tab))
    pd.to_pickle({"names":list(d45["names"]),"table":tab,"failed":[]}, f"{MP}/{name}.pickle"); print(f"{name}: {len(tab)} rows, {miss} missing")
PY
note "descriptors re-assembled: $(grep -E '^descriptors_v1?[0-9c]*' $L | tail -5 | tr '\n' ' ')"
for d in graphs_v4_v7_supercon graphs_v4_v9_cuneg graphs_v4_v11_feox; do rsync -a --delete database/datafiles/MP/$d/ <user>@<cluster-login-host>:/data/<group>/<user>/ML_SC_Proj/database/datafiles/MP/$d/ >> $L 2>&1 || { note "HALT sync $d"; exit 1; }; done
note "HISTFIX SC CHAIN COMPLETE (graphs rebuilt, descriptors re-assembled, synced)"
