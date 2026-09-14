"""Label-consistency sweep: rows whose reported Tc disagrees with their
compositional neighbourhood.

For every row, neighbours = rows of the SAME chemical system within an L1
fractional-composition distance R (default 0.03 = a 3 % change in one
species). A row is an outlier when it has >= 3 neighbours and its Tc sits
further than max(10 K, 50 % of the larger value) from the neighbour median.
Three classes: zero-in-a-high-cloud (a Tc=0 report among superconductors),
high-in-a-zero-cloud (a lone superconductor among non-SC reports), and
mid-outlier (an isolated jump inside a superconducting series).

  python scripts/supercon_expansion/label_outliers.py [--index SC_MP_V10.pickle] [--radius 0.03]
"""
import argparse, os, sys, warnings
from collections import defaultdict
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pymatgen.core import Composition
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MP = os.path.join(_ROOT, "database", "datafiles", "MP")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(MP, "SC_MP_V10.pickle"))
    ap.add_argument("--radius", type=float, default=0.03)
    ap.add_argument("--out", default=os.path.join(_ROOT, "docs", "data_curation", "v10_label_outliers.csv"))
    a = ap.parse_args()
    df = pd.read_pickle(a.index)
    df["formula"] = df.id.map(lambda i: str(i).split("-MP-")[0].split("-ICSD-")[0].replace("-synth_doped", ""))
    v45 = set(pd.read_pickle(os.path.join(MP, "SC_MP_V4_doped_v45.pickle")).id)
    df["origin"] = np.where(df.id.isin(v45), "V45", "new")
    groups = defaultdict(list)
    vec = {}
    for r in df.itertuples():
        try:
            c = Composition(r.formula).fractional_composition.get_el_amt_dict()
        except Exception:  # noqa: BLE001
            continue
        key = "-".join(sorted(c))
        groups[key].append(r.id); vec[r.id] = c
    tc = dict(zip(df.id, df.tc)); form = dict(zip(df.id, df.formula)); orig = dict(zip(df.id, df.origin))
    rows = []
    for key, ids in groups.items():
        if len(ids) < 4:
            continue
        els = key.split("-")
        M = np.array([[vec[i].get(e, 0.0) for e in els] for i in ids])
        D = np.abs(M[:, None, :] - M[None, :, :]).sum(-1)
        for k, i in enumerate(ids):
            nb = [ids[j] for j in np.where((D[k] <= a.radius) & (np.arange(len(ids)) != k))[0]]
            if len(nb) < 3:
                continue
            nt = np.array([tc[j] for j in nb]); med = float(np.median(nt))
            t = tc[i]
            if abs(t - med) <= max(10.0, 0.5 * max(t, med)):
                continue
            jn = nb[int(np.argmin([D[k][ids.index(j)] for j in nb]))]
            cls = ("zero-in-high-cloud" if t == 0 and med > 10 else
                   "high-in-zero-cloud" if med == 0 and t > 10 else "mid-outlier")
            rows.append(dict(id=i, formula=form[i], tc=t, origin=orig[i], cls=cls, n_nbr=len(nb),
                             nbr_median=round(med, 1), nbr_min=float(nt.min()), nbr_max=float(nt.max()),
                             nearest=form[jn], nearest_tc=tc[jn],
                             nearest_dist=round(float(D[k][ids.index(jn)]), 3), chemsys=key))
    out = pd.DataFrame(rows).sort_values(["cls", "tc"], ascending=[True, False])
    out.to_csv(a.out, index=False)
    print(f"rows {len(df)} | in systems with >=4 members {sum(len(v) for v in groups.values() if len(v) >= 4)} "
          f"| flagged {len(out)} ({(out.origin == 'new').sum()} new, {(out.origin == 'V45').sum()} V45) -> {a.out}")
    print(out.cls.value_counts().to_string())


if __name__ == "__main__":
    main()
