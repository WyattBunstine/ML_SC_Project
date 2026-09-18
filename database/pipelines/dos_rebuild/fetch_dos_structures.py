"""DOS-rebuild step 1: fetch structures for the 23,270 DOS materials we lack graphs for,
so DOS can be fetched onto the new +/-1 eV grid for the full 62,972-material universe."""

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)                                     # sibling modules (e.g. synth_dope)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))) # repo root (database.*, models.*)
# 3DSC repo (for synthetic_doping internals); override with THREEDSC_REPO env var.
THREEDSC_REPO = _os.environ.get("THREEDSC_REPO", _os.path.expanduser("~/Downloads/old_files/3DSC-main"))

import os, sys, json, warnings
warnings.filterwarnings("ignore")
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

# Env-only key, per claude.md — never scraped from a local file.


def main():
    key = os.environ.get("MP_API_KEY") or sys.exit("error: set MP_API_KEY (env-only)")
    mids = json.load(open("database/datafiles/MP/dos_rebuild/dos_missing_graphs.json"))
    outdir = "database/datafiles/MP/dos_rebuild/cifs"; os.makedirs(outdir, exist_ok=True)
    done = {f[:-4] for f in os.listdir(outdir) if f.endswith(".cif")}
    todo = [m for m in mids if m not in done]
    print(f"fetching {len(todo)} structures ({len(done)} already present)", flush=True)
    CH = 500; got = 0
    with MPRester(key) as mpr:
        for i in range(0, len(todo), CH):
            chunk = todo[i:i+CH]
            for d in mpr.materials.summary.search(material_ids=chunk, fields=["material_id", "structure"]):
                try:
                    CifWriter(d.structure).write_file(os.path.join(outdir, f"{d.material_id}.cif")); got += 1
                except Exception:
                    pass
            print(f"  {min(i+CH,len(todo))}/{len(todo)} -> {got} cifs", flush=True)
    # build-db source CSV: cif filename + placeholder tc (graphs only need the structure)
    import csv
    with open("database/datafiles/MP/dos_rebuild/dos_new_source.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cif", "tc"])
        for m in mids:
            p = os.path.join(outdir, f"{m}.cif")
            if os.path.exists(p): w.writerow([f"{m}.cif", 0.0])
    print(f"DONE: {got} structures -> {outdir}; wrote dos_new_source.csv", flush=True)


if __name__ == "__main__":
    main()
