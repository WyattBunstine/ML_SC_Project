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
    """Rebuild the pretrained encoder from its checkpoint args through the SAME
    GPSCrystalNet.from_args factory gps_main trains with — so the transfer encoder
    is guaranteed identical to the trained one (no silent architecture drift when a
    new ctor knob is added) and load_state_dict matches key-for-key."""
    from model import GPSCrystalNet
    return GPSCrystalNet.from_args(args, dims)


def embed_index(checkpoint_path, index_path, out_dir, device="cpu", batch_size=64,
                resume=True):
    """Save <id>.npy (per-atom encoder h, (N_atoms, atom_fea_len)) for every structure in
    the index, from a pretrained GPS checkpoint. Resumable (skips existing .npy)."""
    from data import load_cif_dataset, collate_pool_geom
    from torch.utils.data import DataLoader

    os.makedirs(out_dir, exist_ok=True)
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get("args", {})

    # The transfer dataset must produce the SAME feature space the encoder was trained
    # on (the feature flags below come from the checkpoint). target_column is NOT
    # inherited: the pretraining target (e.g. formation_energy_per_atom) is irrelevant
    # to an encoder-only embedding pass and is absent from the transfer index — let the
    # transfer index resolve its own natural target (value/tc) so this never crashes.
    dataset = load_cif_dataset(
        index_path,
        max_num_nbr=args.get("max_num_nbr", 14),
        max_num_poly_nbr=args.get("max_num_poly_nbr", 16),
        target_column=None,
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
        out_paths = [os.path.join(out_dir, str(cid) + ".npy") for cid in cif_ids]
        # Skip the (expensive) encoder forward entirely when every structure in this
        # batch is already embedded — a resumed run shouldn't recompute done batches.
        if resume and all(os.path.exists(p) for p in out_paths):
            continue
        inp = tuple(x.to(device) if torch.is_tensor(x) else x for x in input_batch)
        seg = inp[6].cpu().numpy()
        with torch.no_grad():
            h = model.encode(*inp).float().cpu().numpy()          # (N, atom_fea_len)
        for c, out_path in enumerate(out_paths):
            if resume and os.path.exists(out_path):
                continue
            np.save(out_path, h[seg == c])                        # (n_atoms_c, atom_fea_len)
            done += 1
    print(f"Embedded {done} structures -> {out_dir}/ (per-atom h, dim {model.atom_fea_len})")
    return out_dir
