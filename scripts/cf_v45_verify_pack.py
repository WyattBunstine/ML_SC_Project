"""v4.5 wave gate (CF_SCHEMA 4: symmetric-BA + role-gated bond valence): all
three graph sets uniformly at schema 4 with baked bvs, spot checks that the
BVS FIXES are actually present (planar-O de-biased, IrTe2 zeroed), then pack
under NEW _v45 names and verify headers/dims. Exits nonzero on ANY
inconsistency (wave chain stops before sync)."""
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

assert CF_SCHEMA == 4, f"expected CF_SCHEMA 4, builder has {CF_SCHEMA}"
SETS = [
    "database/datafiles/MP/graphs_v4_doped_v45",
    "database/datafiles/MP/disorder_corpus/graphs_v45",
    "database/datafiles/MP/dos_rebuild/graphs_v4_cf",   # re-augmented in place
]
rng = random.Random(0)
for gdir in SETS:
    paths = glob.glob(os.path.join(gdir, "*.json"))
    if not paths:
        sys.exit(f"NO GRAPHS in {gdir}")
    bad_schema = bad_bvs = 0
    for p in rng.sample(paths, min(400, len(paths))):
        g = json.load(open(p))
        if g.get("cf_schema") != CF_SCHEMA:
            bad_schema += 1
        elif "bvs" not in g["nodes"][0]:
            bad_bvs += 1
    print(f"{gdir}: {len(paths):,} graphs; sampled 400: {bad_schema} off-schema, {bad_bvs} bvs-less")
    if bad_schema or bad_bvs:
        sys.exit(f"{gdir}: not uniformly v4.5")

# spot check 1: LSCO Cu bvs sane + mismatch negative-ish (doped formal > parent
# geometry) — unchanged by the fix (cation side always used the BA row)
g = json.load(open("database/datafiles/MP/graphs_v4_doped_v45/"
                   "Cu1La1.85Sr0.15O4-MP-mp-1077929-synth_doped.cif.json"))
cu = [n for n in g["nodes"] if n["Z"] == 29][0]
assert 1.4 < cu["bvs"] < 2.8, f"LSCO Cu bvs {cu['bvs']} out of band"
assert cu["bvs_mismatch"] < 0.5, f"LSCO doped mismatch {cu['bvs_mismatch']} unexpected"
print(f"LSCO Cu: bvs {cu['bvs']:+.2f}, mismatch {cu['bvs_mismatch']:+.2f} OK")

# spot check 2 (THE FIX, Bug 1): planar O must carry the Cu-side BA weight —
# pre-fix it read bvs -1.13 / mismatch +0.87; post-fix ~ -1.64 / +0.36
o_mis = max(n["bvs_mismatch"] for n in g["nodes"] if n["Z"] == 8)
assert o_mis < 0.55, f"planar O mismatch {o_mis} still pre-fix (~0.87) — BA asymmetry present!"
print(f"LSCO planar O mismatch {o_mis:+.2f} (pre-fix +0.87) — Bug 1 fix present")

# spot check 3 (THE FIX, Bug 2): IrTe2 role/X conflict must zero the block —
# pre-fix Te read +4.3 (mismatch +6.3)
g = json.load(open("database/datafiles/MP/graphs_v4_doped_v45/"
                   "Ir0.93Te2-MP-mp-569322-synth_doped.cif.json"))
mx = max(abs(n["bvs"]) for n in g["nodes"])
assert mx == 0.0, f"IrTe2 max |bvs| {mx} != 0 — role/X gate missing!"
print("IrTe2 bvs all-zero — Bug 2 fix present")

JOBS = [
    ("database/datafiles/MP/SC_MP_V4_doped_v45.pickle",
     "database/datafiles/MP/SC_pack_doped_v45", False, None),
    ("database/datafiles/MP/disorder_corpus/disorder_index_v45.pickle",
     "database/datafiles/MP/disorder_pack_v45", True, "formation_energy_per_atom"),
    ("database/datafiles/MP/dos_rebuild/dos_all_index_cf.pickle",
     "database/datafiles/MP/dos_pack_ef1_v45", True, None),
]
for idx, out, derive, tcol in JOBS:
    shutil.rmtree(out, ignore_errors=True)
    pack_dataset(idx, out, derive_mp_id=derive)
    h = json.load(open(os.path.join(out, "pack_header.json")))
    assert h.get("has_cf") and h.get("has_valence_baked") and h.get("has_bvs"), \
        f"{out}: baked flags incomplete: {h}"
    pk = PackedCIFDataV4(out, target_column=tcol, build_angle_bias=True,
                         use_valence_features=True, use_cf_features=True,
                         use_bvs_features=True)
    dim = pk[0][0][0].shape[-1]
    assert dim == 32, f"{out}: dim {dim} != 32"
    fake = 0
    for i in rng.sample(range(len(pk.data)), 300):
        r = pk._ragged(pk.data[i][2])
        if not np.abs(np.asarray(r["valence"])).sum() > 0:
            fake += 1
    print(f"{out}: n={h['n_samples']:,} dim=32 has_bvs=True fake-zero rows {fake}/300")
    if fake:
        sys.exit(f"{out}: fake-baked rows")
h = json.load(open("database/datafiles/MP/dos_pack_ef1_v45/pack_header.json"))
assert h.get("has_dos"), "dos_pack_ef1_v45 lost the DOS target!"
print("ALL V4.5 PACKS VERIFIED")
