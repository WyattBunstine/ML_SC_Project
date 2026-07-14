"""Phase-1 index assembly: V4M / V6M = base + magnetic tc=0 hard negatives.

Each matched magnetic composition becomes a row (label=1, tc=0 — the same
pattern as the existing non-SC parents in the regression pool, so FineTune
wiring is untouched), pointing at its matched MP structure's cgv4 graph
(existing SC/DOS-rebuild graphs reused; freshly built ones under
magnetic_rebuild/graphs_v4). Ordering type + T_order ride along as extra
columns for the phase-2 ground-state classifier.

Also REGENERATES the family-holdout lists for the augmented indexes: any new
magnetic member of the nickelate (Ni-active oxide, no Cu) family or the
Cu1La2 (LSCO) family must be forced into test with its family, or it would
leak as a trainable near-duplicate of held-out rows.

Outputs (database/datafiles/MP/):
  SC_MP_V4M.pickle / SC_MP_V6M.pickle
  descriptors_v4m.pickle / descriptors_v6m.pickle  (base tables + new rows)
  nickelate_holdout_ids_v4m.csv / _v6m.csv, lsco_holdout_ids_v4m.csv / _v6m.csv
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
_ROOT = _os.path.dirname(_os.path.dirname(_HERE))
_sys.path.insert(0, _ROOT)

import os
import re

import numpy as np
import pandas as pd
from pymatgen.core import Composition

MP = "database/datafiles/MP"
GRAPH_DIRS = [f"{MP}/graphs_v4", f"{MP}/dos_rebuild/graphs_v4",
              f"{MP}/magnetic_rebuild/graphs_v4"]


def _fmt(v):
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _mk_id(key, mid):
    """'La1Ni1O3-MP-mp-19339.cif'-style id: alphabetical element-amount prefix
    (the repo convention parent_composition_groups/chemsys_groups parse). Built
    from the canonical composition KEY ('El1.5|El2|...'), not the raw formula
    string — some NEMAD formula spellings don't re-parse (e.g. 'Fe3-xO4')."""
    parts = {ev.rstrip("0123456789."): float(re.sub(r"^[A-Za-z]+", "", ev))
             for ev in str(key).split("|")}
    pre = "".join(f"{e}{_fmt(parts[e])}" for e in sorted(parts))
    return f"{pre}-MP-{mid}.cif"


def main():
    matches = pd.read_csv("database/datafiles/NE_SCDB/magnetic_matches.csv")
    have = {}
    for gd in GRAPH_DIRS:
        if os.path.isdir(gd):
            for f in os.listdir(gd):
                if f.endswith(".cif.json") and f.startswith("mp-"):
                    have.setdefault(f[:-9], os.path.join(gd, f))

    rows, no_graph = [], 0
    for r in matches.itertuples():
        gp = have.get(r.material_id)
        if gp is None:
            no_graph += 1
            continue
        rows.append(dict(id=_mk_id(r.key, r.material_id), value=0.0, graph_path=gp,
                         label=1, tc=0.0, weight=min(1.0, 0.3 + 0.1 * r.n_reports),
                         source="nemad_mag", mag_order=r.order, T_order=r.T_order))
    mag = pd.DataFrame(rows).drop_duplicates("id")
    print(f"magnetic rows with graphs: {len(mag)} (no graph: {no_graph})")

    from models.head.descriptors import build_descriptor_table
    from models.head.HeadData import parent_composition_groups

    tmp_idx = f"{MP}/magnetic_rows_tmp.pickle"
    mag.to_pickle(tmp_idx)
    build_descriptor_table(tmp_idx, f"{MP}/descriptors_magnetic_rows.pickle")
    mdesc = pd.read_pickle(f"{MP}/descriptors_magnetic_rows.pickle")

    for tag, base_idx, base_desc in (
            ("v4m", f"{MP}/SC_MP_V4_doped.pickle", f"{MP}/descriptors_doped.pickle"),
            ("v6m", f"{MP}/SC_MP_V6_expanded.pickle", f"{MP}/descriptors_v6_expanded.pickle")):
        base = pd.read_pickle(base_idx)
        add = mag[~mag.id.isin(set(base.id.astype(str)))].copy()
        # drop rows whose descriptors failed (assemble would drop them unevenly)
        add = add[add.id.isin(mdesc["table"].keys())]
        merged = pd.concat([base, add], ignore_index=True)
        merged.to_pickle(f"{MP}/SC_MP_{tag.upper()}.pickle")
        bd = pd.read_pickle(base_desc)
        table = dict(bd["table"])
        table.update({k: v for k, v in mdesc["table"].items() if k in set(add.id)})
        pd.to_pickle({"names": bd["names"], "table": table,
                      "failed": list(bd.get("failed", []))}, f"{MP}/descriptors_{tag}.pickle")
        print(f"{tag.upper()}: {len(base)} base + {len(add)} magnetic = {len(merged)} rows; "
              f"descriptors {len(table)}")

        # ---- regenerate family holdouts on the augmented index ----
        ids = merged.id.astype(str).tolist()
        grp = parent_composition_groups(ids)
        tc = merged.tc.to_numpy()
        # LSCO: every Cu1La2 group member
        lsco = merged[[g == "Cu1La2" for g in grp]]
        pd.DataFrame({"id": lsco.id.astype(str),
                      "formula": lsco.id.astype(str).str.split("-MP-").str[0],
                      "tc": lsco.tc, "source": f"lsco_family_{tag}"}) \
            .to_csv(f"{MP}/lsco_holdout_ids_{tag}.csv", index=False)
        # Nickelate: the CURATED baseline 43 (unchanged, so M-arm results compare
        # 1:1 with the V4/V6 baselines) UNION any NEW magnetic Ni-oxide rows —
        # which would otherwise train as near-duplicates of the held-out family.
        def is_ni_oxide(cid):
            try:
                els = {e.symbol for e in Composition(
                    re.split(r"-MP-|-ICSD-", str(cid))[0]).elements}
            except Exception:  # noqa: BLE001
                return False
            # exclude pnictide/phosphide 1111s (Ni-As/Ni-P active layers, ≤~9 K)
            return ({"Ni", "O"} <= els and "Cu" not in els
                    and not els & {"As", "P", "Sb", "B", "C"})
        base43 = pd.read_csv(f"{MP}/nickelate_holdout_ids.csv")
        new_mag = merged[(merged.get("source") == "nemad_mag").to_numpy()
                         & np.array([is_ni_oxide(c) for c in merged.id.astype(str)])]
        nick = pd.concat([
            base43[["id", "formula", "tc"]].assign(source="baseline43"),
            pd.DataFrame({"id": new_mag.id.astype(str),
                          "formula": new_mag.id.astype(str).str.split("-MP-").str[0],
                          "tc": new_mag.tc, "source": "nemad_mag"}),
        ]).drop_duplicates("id")
        nick = nick[nick.id.isin(set(merged.id.astype(str)))]
        nick.to_csv(f"{MP}/nickelate_holdout_ids_{tag}.csv", index=False)
        print(f"  holdouts: LSCO {len(lsco)} ({int((lsco.tc > 0).sum())} SC), "
              f"nickelate {len(nick)} (baseline43 + "
              f"{int((nick.source == 'nemad_mag').sum())} new magnetic)")
    os.remove(tmp_idx)


if __name__ == "__main__":
    main()
