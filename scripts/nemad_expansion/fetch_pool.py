"""Fetch MP crystal pool (lightweight fields) for all needed chemical systems,
for matching NEMAD candidates to parent structures. Structures fetched later
only for selected best-match parents."""

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)                                     # sibling modules (e.g. synth_dope)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_HERE))) # repo root (database.*, models.*)
# 3DSC repo (for synthetic_doping internals); override with THREEDSC_REPO env var.
THREEDSC_REPO = _os.environ.get("THREEDSC_REPO", _os.path.expanduser("~/Downloads/old_files/3DSC-main"))

import os, sys, pandas as pd
# Env-only key, per claude.md ("read from the MP_API_KEY environment variable, no
# longer hardcoded") — never scraped from a local file.
key = os.environ.get("MP_API_KEY") or sys.exit("error: set MP_API_KEY (env-only)")
from mp_api.client import MPRester

# argv: [systems_file] [out_pickle] — defaults are the original SC-expansion paths.
SYSTEMS = sys.argv[1] if len(sys.argv) > 1 else "database/datafiles/NE_SCDB/need_systems.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "database/datafiles/NE_SCDB/mp_crystal_pool.pickle"
systems=[l.strip() for l in open(SYSTEMS) if l.strip()]
print(f"fetching {len(systems)} chemical systems -> {OUT}", flush=True)
CH=200
rows=[]
with MPRester(key) as mpr:
    for i in range(0,len(systems),CH):
        chunk=systems[i:i+CH]
        docs=mpr.materials.summary.search(chemsys=chunk,
                fields=["material_id","formula_pretty","composition_reduced","energy_above_hull"])
        for d in docs:
            cr=d.composition_reduced
            cd={str(el):float(amt) for el,amt in (cr.items() if hasattr(cr,"items") else dict(cr).items())}
            rows.append(dict(material_id=str(d.material_id),formula_pretty=str(d.formula_pretty),
                             reduced=cd, eah=float(d.energy_above_hull) if d.energy_above_hull is not None else 9.9))
        print(f"  {i+len(chunk)}/{len(systems)} systems -> {len(rows)} materials", flush=True)
df=pd.DataFrame(rows).drop_duplicates("material_id")
df.to_pickle(OUT)
print(f"DONE: {len(df)} unique MP materials -> {OUT}", flush=True)
