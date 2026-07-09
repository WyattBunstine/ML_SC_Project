"""End-to-end smoke: the real DOS pack consumed exactly as gps_main rung-04 would —
multitask dataset -> collate_pool_multitask -> GPSCrystalNet.from_args forward (all heads,
autograd forces/stress) -> _mt_loss. Confirms the dos target/mask flow through and the dos
head produces a finite per-structure spectrum, on a positioned, exact-to_jimage pack."""

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)                                     # sibling modules (e.g. synth_dope)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_HERE))) # repo root (database.*, models.*)
# 3DSC repo (for synthetic_doping internals); override with THREEDSC_REPO env var.
THREEDSC_REPO = _os.environ.get("THREEDSC_REPO", _os.path.expanduser("~/Downloads/old_files/3DSC-main"))

import json, os, sys
sys.path.insert(0, "models/common")
sys.path.insert(0, "models/GPSTransformer")
import torch
from data import load_cif_dataset, collate_pool_multitask
from train import _build_cart_strain, _mt_loss, compute_target_stats
from model import GPSCrystalNet

cfg = json.load(open("configs/gps_mt_ablation_suite/04_dos_full.json"))
PACK = "database/datafiles/MP/dos_pack"

ds = load_cif_dataset(
    PACK,
    max_num_nbr=cfg["max_num_nbr"], max_num_poly_nbr=cfg["max_num_poly_nbr"],
    target_column=cfg["target_column"],
    use_bond_angles=cfg.get("use_bond_angles", False),
    use_poly_edges=cfg["use_poly_edges"], build_angle_bias=True,
    use_rich_node_features=cfg.get("use_rich_node_features", False),
    use_dihedrals=cfg.get("use_dihedrals", False),
    multitask=True,
)
print(f"dataset: {len(ds)} samples (multitask)")

# Grab a spread of samples likely to include some with a real DOS label.
idxs = list(range(0, 24000, 1000))[:24]
batch = [ds[i] for i in idxs]
input_batch, targets, masks, cif_ids = collate_pool_multitask(batch)

sa, sn, _, sp, _, _ = ds[0][0][:6]
model = GPSCrystalNet.from_args(cfg, (sa.shape[-1], sn.shape[-1], sp.shape[-1]))
model.train()

input_var = list(input_batch)  # cpu
cart, strain = _build_cart_strain(input_var)
out = model(*input_var, cart=cart, strain=strain)

print("output heads:", sorted(out.keys()))
dos = out["dos"]
print(f"dos out: shape={tuple(dos.shape)}  finite={torch.isfinite(dos).all().item()}  "
      f"min={dos.min().item():.3g} max={dos.max().item():.3g}")
dmask = masks["dos"]
print(f"dos mask in batch: {int(dmask.sum().item())}/{len(idxs)} samples carry a real DOS target")
print(f"forces head present & finite: {torch.isfinite(out['forces']).all().item()} "
      f"(force targets all masked: mask sum={int(masks['forces'].sum().item())})")

# Loss must be finite and backprop must reach params (the real training step).
stats = compute_target_stats(ds, list(range(min(3000, len(ds)))))
loss, maes = _mt_loss(out, targets, masks, stats, cfg["loss_weights"], input_var[6])
loss.backward()
gnorm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
print(f"loss={loss.item():.4f}  finite={torch.isfinite(loss).item()}  per-task MAE={ {k: round(float(v),3) for k,v in maes.items()} }")
print(f"total |param grad| = {gnorm:.3g}  -> {'OK: grads flow' if gnorm>0 else 'FAIL: no grad'}")
print("SMOKE PASS" if (torch.isfinite(loss).item() and gnorm > 0 and int(dmask.sum())>0) else "SMOKE FAIL")
