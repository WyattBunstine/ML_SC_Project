"""Schema-2 wave step 4: verify all three CF graph sets are uniformly stamped,
spot-check the d9 physics, then repack (fresh dirs) and verify the packs.
Exits nonzero on ANY inconsistency — the wave chain stops before syncing."""
import glob
import json
import os
import random
import shutil
import sys

import numpy as np

sys.path.insert(0, "models/common")
sys.path.insert(0, "database")
import crystal_graph_v4_import  # noqa: F401,E402
from crystal_field_aom import CF_SCHEMA  # noqa: E402
from pack import PackedCIFDataV4, pack_dataset  # noqa: E402

SETS = [
    ("database/datafiles/MP/graphs_v4_doped_cf", None),
    ("database/datafiles/MP/disorder_corpus/graphs_v4_cf", None),
    ("database/datafiles/MP/dos_rebuild/graphs_v4_cf", None),
]
rng = random.Random(0)
for gdir, _ in SETS:
    paths = glob.glob(os.path.join(gdir, "*.json"))
    if not paths:
        sys.exit(f"NO GRAPHS in {gdir}")
    bad = 0
    for p in rng.sample(paths, min(400, len(paths))):
        if json.load(open(p)).get("cf_schema") != CF_SCHEMA:
            bad += 1
    print(f"{gdir}: {len(paths):,} graphs, sampled 400, {bad} off-schema")
    if bad:
        sys.exit(f"{gdir}: {bad} graphs not at cf_schema={CF_SCHEMA}")

# d9 spot check on the SC set (LSCO)
g = json.load(open("database/datafiles/MP/graphs_v4_doped_cf/"
                   "Cu1La1.85Sr0.15O4-MP-mp-1077929-synth_doped.cif.json"))
cu = [n for n in g["nodes"] if n["Z"] == 29][0]
occ_sum = sum(cu["cf"][5:10])
assert abs(occ_sum - 8.85) < 1e-2, f"LSCO Cu d-count {occ_sum} != 8.85"
print(f"LSCO d9 spot check OK (occ sum {occ_sum:.3f}, gap {cu['cf'][10]:.2f})")

# repack (fresh dirs — never in place)
JOBS = [
    ("database/datafiles/MP/SC_MP_V4_doped_cf.pickle",
     "database/datafiles/MP/SC_pack_doped_cf", False, None),
    ("database/datafiles/MP/disorder_corpus/disorder_index_cf.pickle",
     "database/datafiles/MP/disorder_pack_cf", True, "formation_energy_per_atom"),
    ("database/datafiles/MP/dos_rebuild/dos_all_index_cf.pickle",
     "database/datafiles/MP/dos_pack_ef1_cf2", True, None),
]
for idx, out, derive, tcol in JOBS:
    shutil.rmtree(out, ignore_errors=True)
    pack_dataset(idx, out, derive_mp_id=derive)
    h = json.load(open(os.path.join(out, "pack_header.json")))
    assert h.get("has_cf") and h.get("has_valence_baked"), f"{out}: missing baked flags"
    pk = PackedCIFDataV4(out, target_column=tcol, build_angle_bias=True)
    fake = 0
    for i in rng.sample(range(len(pk.data)), 300):
        r = pk._ragged(pk.data[i][2])
        if not np.abs(np.asarray(r["valence"])).sum() > 0:
            fake += 1
    print(f"{out}: n={h['n_samples']:,} has_cf={h['has_cf']} fake-zero rows {fake}/300")
    if fake:
        sys.exit(f"{out}: {fake} fake-baked rows — corrupted")
# DOS pack must additionally carry the dos target
h = json.load(open("database/datafiles/MP/dos_pack_ef1_cf2/pack_header.json"))
assert h.get("has_dos"), "dos_pack_ef1_cf2 lost the DOS target!"
print("ALL SCHEMA-2 PACKS VERIFIED")
