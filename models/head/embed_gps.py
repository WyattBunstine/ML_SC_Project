"""Export per-structure GPS encoder embeddings for the frozen T_c transfer probe.

The twin of embed_mace.py: instead of MACE descriptors, run a PRETRAINED multitask
GPSCrystalNet encoder over the transfer structures and save each structure's per-atom
representation h (shape (N_atoms, atom_fea_len)) as <id>.npy — the SAME per-structure
layout embed_mace writes, so HeadData.assemble pools (mean+max) and the existing TcHead
pipeline (HeadMain) consumes them UNCHANGED.

    python main.py embed-gps --checkpoint <pretrained.pth.tar> \
        --index database/datafiles/MP/SC_MP_V4.pickle --out embeddings/gps_pretrained

Uses model.encode() = the STATIC-feature encoder path (no Cartesian leaf / autograd),
which equals the differentiable recompute at the reference geometry — so the frozen
embeddings need only a forward pass. The transfer dataset is built with the SAME feature
flags the encoder was trained with (read from the checkpoint's args).
"""
import os

import numpy as np
import torch


def _build_model_from_args(args, dims):
    from model import GPSCrystalNet
    oa, nb, po = dims
    return GPSCrystalNet(
        oa, nb, poly_fea_len=po,
        atom_fea_len=args.get("atom_feat_len", 128), n_conv=args.get("n_conv", 4),
        h_fea_len=args.get("h_feat_len", 128), n_h=args.get("n_hidden", 2),
        use_poly_edges=args.get("use_poly_edges", True),
        atom_pooling=args.get("atom_pooling", "mean"),
        n_heads=args.get("set_transformer_heads", 8),
        gps_global=args.get("gps_global", True),
        gps_global_heads=args.get("gps_global_heads", args.get("set_transformer_heads", 8)),
        gps_ffn_mult=args.get("gps_ffn_mult", 2),
        local_transformer=args.get("local_transformer", True),
        per_atom_head=args.get("per_atom_head", True),
        use_bond_edges=args.get("use_bond_edges", True),
        shell_aggregation=args.get("shell_aggregation", "attention"),
        use_angle_bias=args.get("use_angle_bias", True),
        use_dist_bias=args.get("use_dist_bias", False),
        dist_cutoff=args.get("dist_cutoff", 8.0), n_dist_rbf=args.get("n_dist_rbf", 16),
        tasks=(set(args["tasks"]) if args.get("tasks") else None),
        differentiable_geometry=bool(args.get("differentiable_geometry")),
        n_energy=args.get("n_energy", 256))


def embed_index(checkpoint_path, index_path, out_dir, device="cpu", batch_size=64,
                resume=True):
    """Save <id>.npy (per-atom encoder h, (N_atoms, atom_fea_len)) for every structure in
    the index, from a pretrained GPS checkpoint. Resumable (skips existing .npy)."""
    from data import load_cif_dataset, collate_pool_geom
    from torch.utils.data import DataLoader

    os.makedirs(out_dir, exist_ok=True)
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get("args", {})

    # The transfer dataset must produce the SAME feature space the encoder was trained on.
    dataset = load_cif_dataset(
        index_path,
        max_num_nbr=args.get("max_num_nbr", 14),
        max_num_poly_nbr=args.get("max_num_poly_nbr", 16),
        target_column=args.get("target_column"),
        use_poly_edges=args.get("use_poly_edges", True),
        use_bond_angles=args.get("use_bond_angles", False),
        build_angle_bias=True,
        use_rich_node_features=args.get("use_rich_node_features", False),
        use_dihedrals=args.get("use_dihedrals", False))

    sa, sn, _, sp, _, _ = dataset[0][0][:6]
    model = _build_model_from_args(args, (sa.shape[-1], sn.shape[-1], sp.shape[-1]))
    model.load_state_dict(ckpt["state_dict"])     # restores weights + feature-stat buffers
    model.eval().to(device)

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_pool_geom)
    done = 0
    for input_batch, _t, _l, cif_ids in loader:
        inp = tuple(x.to(device) if torch.is_tensor(x) else x for x in input_batch)
        seg = inp[6].cpu().numpy()
        with torch.no_grad():
            h = model.encode(*inp).float().cpu().numpy()          # (N, atom_fea_len)
        for c, cid in enumerate(cif_ids):
            out_path = os.path.join(out_dir, str(cid) + ".npy")
            if resume and os.path.exists(out_path):
                continue
            np.save(out_path, h[seg == c])                        # (n_atoms_c, atom_fea_len)
            done += 1
    print(f"Embedded {done} structures -> {out_dir}/ (per-atom h, dim {model.atom_fea_len})")
    return out_dir
