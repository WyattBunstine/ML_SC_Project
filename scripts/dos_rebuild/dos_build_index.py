"""DOS-rebuild step 3: build a unified index of all DOS materials pointing at their
UNDOPED (mp-id-named) graphs, for fetch-dos to attach the new +/-1 eV DOS onto."""

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)                                     # sibling modules (e.g. synth_dope)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_HERE))) # repo root (database.*, models.*)
# 3DSC repo (for synthetic_doping internals); override with THREEDSC_REPO env var.
THREEDSC_REPO = _os.environ.get("THREEDSC_REPO", _os.path.expanduser("~/Downloads/old_files/3DSC-main"))

import json, glob, os, re, pandas as pd

dos_ids = set(json.load(open("database/datafiles/MP/dos_rebuild/dos_material_ids.json")))
# undoped MP graphs = files named purely by mp-id (mp-123.cif.json or mp-123.json), across
# MP_Energy + the freshly-built dos_rebuild. Exclude doped SC graphs (Formula-MP-mp-...).
mid2graph = {}
for gd in glob.glob("database/datafiles/*/graphs_v4*") + ["database/datafiles/MP/dos_rebuild/graphs_v4"]:
    if not os.path.isdir(gd):
        continue
    for f in glob.glob(os.path.join(gd, "*.json")):
        b = os.path.basename(f)
        m = re.match(r"^(mp-\d+)", b)                # undoped only
        if m and m.group(1) in dos_ids and m.group(1) not in mid2graph:
            mid2graph[m.group(1)] = f
rows = [{"id": mid, "graph_path": gp, "tc": 0.0} for mid, gp in sorted(mid2graph.items())]
df = pd.DataFrame(rows)
out = "database/datafiles/MP/dos_rebuild/dos_all_index.pickle"
df.to_pickle(out)
df.to_csv(out.replace(".pickle", ".csv"), index=False)
print(f"DOS index: {len(df)} materials with undoped graphs -> {out}")
print(f"  (of {len(dos_ids)} MP DOS materials; {len(dos_ids)-len(df)} have no undoped graph)")
# sanity: graph_paths resolve
miss = sum(not os.path.exists(r.graph_path) for r in df.head(200).itertuples())
print(f"  first-200 graph_path existence check: {200-miss}/200 present")
