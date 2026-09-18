"""V10 = V9 minus the audited hard exclusions (docs/data_curation/v9_audit_exclude_ids.csv
+ v45_audit_exclude_ids.csv), with descriptors, family metadata and every holdout
list carried over. The review list is KEPT in (it is a watch list, not an exclusion).
  python database/pipelines/supercon_expansion/finalize_v10.py
"""
import os, sys
import pandas as pd
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MP = os.path.join(_ROOT, "database", "datafiles", "MP"); DC = os.path.join(_ROOT, "docs", "data_curation")


def main():
    v9 = pd.read_pickle(os.path.join(MP, "SC_MP_V9_cuneg.pickle"))
    ex = set(pd.read_csv(os.path.join(DC, "v9_audit_exclude_ids.csv")).id)
    p45 = os.path.join(DC, "v45_audit_exclude_ids.csv")
    if os.path.exists(p45):
        ex |= set(pd.read_csv(p45).id)
    # label conflicts: identical nominal compositions with >= 3 reports where the
    # minority class (zero vs superconducting) is outvoted (user-approved 2026-09-14)
    pmv = os.path.join(DC, "v10_conflict_minority_ids.csv")
    if os.path.exists(pmv):
        mv = set(pd.read_csv(pmv).id); ex |= mv
        print(f"  majority-vote label removals: {len(mv)}")
    # 1-vs-1 zero-vs-SC conflicts: no majority exists, both sides dropped (user decision 2026-09-14)
    ppr = os.path.join(DC, "v10_conflict_pairs_drop_ids.csv")
    if os.path.exists(ppr):
        pr = set(pd.read_csv(ppr).id); ex |= pr
        print(f"  unresolved-pair removals: {len(pr)}")
    v10 = v9[~v9.id.isin(ex)].reset_index(drop=True)
    v10.to_pickle(os.path.join(MP, "SC_MP_V10.pickle")); v10.to_csv(os.path.join(MP, "SC_MP_V10.csv"), index=False)
    d9 = pd.read_pickle(os.path.join(MP, "descriptors_v9_cuneg.pickle"))
    keep = set(v10.id)
    pd.to_pickle({"names": d9["names"], "table": {k: v for k, v in d9["table"].items() if k in keep}, "failed": []},
                 os.path.join(MP, "descriptors_v10.pickle"))
    meta = pd.read_csv(os.path.join(MP, "3DSC_MP_v9.csv"), low_memory=False)
    meta.to_csv(os.path.join(MP, "3DSC_MP_v10.csv"), index=False)
    n_h = 0
    for src, dst in (("la_series_both_holdout_ids_v9.csv", "la_series_both_holdout_ids_v10.csv"),
                     ("nickelate_holdout_ids_v45_v9.csv", "nickelate_holdout_ids_v10.csv"),
                     ("holdout_matthias_dome1_ids.csv", "holdout_matthias_dome1_ids_v10.csv"),
                     ("holdout_matthias_dome2_ids.csv", "holdout_matthias_dome2_ids_v10.csv"),
                     ("holdout_matthias_more_ids.csv", "holdout_matthias_more_ids_v10.csv"),
                     ("holdout_matthias_nbzr_ids.csv", "holdout_matthias_nbzr_ids_v10.csv")):
        p = os.path.join(MP, src)
        if not os.path.exists(p):
            continue
        h = pd.read_csv(p); h2 = h[h.id.isin(keep)]
        h2.to_csv(os.path.join(MP, dst), index=False); n_h += 1
        print(f"  {dst}: {len(h2)} ids ({len(h) - len(h2)} dropped as excluded)")
    cu = v10[v10.id.str.split("-MP-").str[0].str.contains("Cu") & v10.id.str.split("-MP-").str[0].str.contains("O")]
    print(f"V10: {len(v10)} rows (V9 {len(v9)} - {len(v9) - len(v10)} excluded) | Tc>0 {int((v10.tc > 0).sum())} "
          f"({100 * (v10.tc > 0).mean():.1f}%) | ratio {(v10.tc > 0).sum() / (v10.tc <= 0).sum():.2f}:1 | "
          f"Cu-O rows {len(cu)} ({int((cu.tc <= 0).sum())} zeros) | descriptors {len(keep)} | {n_h} holdout lists")


if __name__ == "__main__":
    main()
