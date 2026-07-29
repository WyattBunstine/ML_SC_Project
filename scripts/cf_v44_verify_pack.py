"""v4.4 wave gate: all three graph sets uniformly at CF_SCHEMA 3 with baked
bvs, spot checks, then pack under NEW _v44 names and verify headers/dims.
Exits nonzero on ANY inconsistency (wave chain stops before sync)."""
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

assert CF_SCHEMA == 3, f"expected CF_SCHEMA 3, builder has {CF_SCHEMA}"
SETS = [
    "database/datafiles/MP/graphs_v4_doped_v44",
    "database/datafiles/MP/disorder_corpus/graphs_v44",
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
        sys.exit(f"{gdir}: not uniformly v4.4")

# spot check: LSCO Cu bvs sane + mismatch negative-ish (doped formal > parent-geometry bvs)
g = json.load(open("database/datafiles/MP/graphs_v4_doped_v44/"
                   "Cu1La1.85Sr0.15O4-MP-mp-1077929-synth_doped.cif.json"))
cu = [n for n in g["nodes"] if n["Z"] == 29][0]
assert 1.4 < cu["bvs"] < 2.8, f"LSCO Cu bvs {cu['bvs']} out of band"
assert cu["bvs_mismatch"] < 0.5, f"LSCO doped mismatch {cu['bvs_mismatch']} unexpected"
print(f"LSCO Cu: bvs {cu['bvs']:+.2f}, mismatch {cu['bvs_mismatch']:+.2f} "
      f"(doped formal {cu['oxidation_state']:+.2f} vs parent geometry) OK")

JOBS = [
    ("database/datafiles/MP/SC_MP_V4_doped_v44.pickle",
     "database/datafiles/MP/SC_pack_doped_v44", False, None),
    ("database/datafiles/MP/disorder_corpus/disorder_index_v44.pickle",
     "database/datafiles/MP/disorder_pack_v44", True, "formation_energy_per_atom"),
    ("database/datafiles/MP/dos_rebuild/dos_all_index_cf.pickle",
     "database/datafiles/MP/dos_pack_ef1_v44", True, None),
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
h = json.load(open("database/datafiles/MP/dos_pack_ef1_v44/pack_header.json"))
assert h.get("has_dos"), "dos_pack_ef1_v44 lost the DOS target!"
print("ALL V4.4 PACKS VERIFIED")
