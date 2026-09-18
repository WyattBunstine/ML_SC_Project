"""Val-split parity dumps for the formation-energy ladder (rungs 43/44/45/27/46).

The pretrains never save predictions, so this rebuilds each rung's dataset +
split EXACTLY as gps_main does (same _ds_kw enumeration, same seed/split_by ->
identical val membership) and dumps (cif_id, true, pred) for every val row with
an energy label. Runs where the packs live (cluster scratch).

Usage: python scripts/fe_parity.py   -> model_data/fe_parity/<rung>.csv
"""
import csv
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
for _p in ("models/common", "models/GPSTransformer"):
    sys.path.insert(0, os.path.join(_ROOT, _p))

from data import (load_cif_dataset, get_sc_nonsc_loaders, resolve_split_by,  # noqa: E402
                  collate_pool_multitask, ConcatMTDataset)
from model import GPSCrystalNet  # noqa: E402

RUNGS = {
    "43_element": "model_data/2026-09-03/gps_mt_43_fe_element/gps_mt_43_fe_element_2026-09-03_11-18-19/result_t43_model_best.pth.tar",
    "44_node":    "model_data/2026-09-03/gps_mt_44_fe_node/gps_mt_44_fe_node_2026-09-03_12-35-23/result_t44_model_best.pth.tar",
    "45_bonds":   "model_data/2026-09-03/gps_mt_45_fe_bonds/gps_mt_45_fe_bonds_2026-09-03_14-03-22/result_t45_model_best.pth.tar",
    "27_angles":  "model_data/2026-08-25/gps_mt_27_t_energy/gps_mt_27_t_energy_2026-08-25_10-14-08/result_t27_model_best.pth.tar",
    "46_poly":    "model_data/2026-09-03/gps_mt_46_fe_poly/gps_mt_46_fe_poly_2026-09-03_14-25-42/result_t46_model_best.pth.tar",
    "50_full":    "model_data/2026-09-08/gps_mt_50_fe_full/gps_mt_50_fe_full_2026-09-08_11-36-24/result_t50_model_best.pth.tar",
}


def build_val_loader(args):
    ds_kw = dict(
        max_num_nbr=args.get("max_num_nbr", 14),
        max_num_poly_nbr=args.get("max_num_poly_nbr", 16),
        graph_cache_size=args.get("graph_cache_size", 4096),
        target_column=args.get("target_column"),
        use_bond_angles=args.get("use_bond_angles", False),
        use_poly_edges=args.get("use_poly_edges", True),
        build_angle_bias=True,
        use_rich_node_features=args.get("use_rich_node_features", False),
        use_valence_features=args.get("use_valence_features", False),
        use_cf_features=args.get("use_cf_features", False),
        use_bvs_features=args.get("use_bvs_features", False),
        mask_oxidation_feature=args.get("mask_oxidation_feature", False),
        mask_geometry_features=args.get("mask_geometry_features", False),
        use_poly_node_summary=args.get("use_poly_node_summary", False),
        use_dihedrals=args.get("use_dihedrals", False),
        multitask=True,
        n_energy=args.get("n_energy", 256),
        dos_per_atom=args.get("dos_per_atom", True),
        frame_subsample=args.get("frame_subsample", 1),
    )
    ips = args["index_path"] if isinstance(args["index_path"], list) else [args["index_path"]]
    subsets = [load_cif_dataset(ip, **ds_kw) for ip in ips]
    dataset = subsets[0] if len(subsets) == 1 else ConcatMTDataset(subsets)
    loaders = get_sc_nonsc_loaders(
        dataset, batch_size=args["batch_size"],
        val_ratio=args["val_ratio"], test_ratio=args["test_ratio"],
        sc_to_nonsc_ratio=float("inf"), num_workers=8, pin_memory=True,
        seed=args.get("split_seed", 123),
        split_by=resolve_split_by(args.get("split_by"), dataset),
        size_grouped=args.get("size_grouped_batches", False),
        max_atoms_per_batch=args.get("max_atoms_per_batch"),
        size_pool_factor=args.get("size_pool_factor", 20),
        collate_fn=collate_pool_multitask)
    return dataset, loaders["val_realistic"]


def main():
    dev = torch.device("cuda")
    os.makedirs("model_data/fe_parity", exist_ok=True)
    only = set(sys.argv[1:])                      # optional: which rung tags to dump
    for tag, ckpt in RUNGS.items():
        if only and tag not in only:
            continue
        if os.path.exists(f"model_data/fe_parity/{tag}.csv") and not only:
            print(f"{tag}: exists, skipping", flush=True); continue
        ck = torch.load(ckpt, map_location=dev)
        args = ck["args"]
        dataset, val = build_val_loader(args)
        s0 = dataset[0][0]
        model = GPSCrystalNet.from_args(args, (s0[0].shape[-1], s0[1].shape[-1], s0[3].shape[-1]))
        model.load_state_dict(ck["state_dict"])
        model = model.to(dev).eval()
        rows = []
        with torch.no_grad():
            for inp, targets, masks, cids in val:
                inp = tuple(x.to(dev) if torch.is_tensor(x) else x for x in inp)
                out = model(*inp)              # cart=None -> energy only, no autograd
                m = masks["energy"].bool()
                e_p = out["energy"].detach().cpu()
                e_t = targets["energy"]
                for i, cid in enumerate(cids):
                    if m[i]:
                        rows.append((str(cid), float(e_t[i]), float(e_p[i])))
        with open(f"model_data/fe_parity/{tag}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "true_eV_atom", "pred_eV_atom"])
            w.writerows(rows)
        mae = float(np.mean([abs(t - p) for _, t, p in rows]))
        print(f"{tag}: {len(rows)} val rows, MAE {mae:.4f} eV/atom", flush=True)
        del model, dataset, val


if __name__ == "__main__":
    main()
