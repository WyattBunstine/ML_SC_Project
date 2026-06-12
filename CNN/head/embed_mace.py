"""One-time MACE embedding pass: frozen per-atom encoder features for the T_c head.

For every row of a cgv4 index this computes the pretrained MACE descriptor matrix
(N_atoms, D) for the structure and writes one float32 ``<id>.npy`` into the output
directory, plus a ``manifest.json`` (model, dim, settings) and ``failed.txt``.

Atom-order alignment is the correctness crux: the head later joins these rows
with graph-derived features, so the i-th embedding row must be the i-th graph
node. We guarantee that by construction *and* by assertion:

- construction: the CIF is parsed with the exact same loader the graph builder
  uses (``_load_unit_cell_structure_from_cif``: full cell, no primitive
  reduction, no symmetrization), and disordered sites are collapsed with the
  builder's own ``_dominant_symbol`` rule (max occupancy, alphabetical
  tie-break) — both imported from RPToleranceFactor, not re-implemented;
- assertion: every structure's per-site Z sequence is compared against the
  stored graph JSON's node Z list, and any mismatch fails that structure
  loudly into failed.txt (never written with silent misalignment).

MACE features are extracted with ``MACECalculator.get_descriptors(...,
invariants_only=True)``: the rotation-invariant (l=0) channels of the node
features after each interaction layer, concatenated — the standard "MLIP as
feature provider" representation.
"""

import json
import os
import time

import numpy as np

# Resolving RPToleranceFactor on sys.path (same convention as the DB builder).
from database.crystal_graph_v4_import import build_crystal_graph_from_cif  # noqa: F401
from crystal_graph_v4 import (  # type: ignore  # resolved by the import above
    _dominant_symbol,
    _get_site_species_info,
    _load_unit_cell_structure_from_cif,
)

VACANCY_SYMBOLS = {"X", "X0+"}  # builder's vacancy placeholder can't be embedded


def remap_graph_path(path, prefix_map):
    """Apply an 'old:new' path-prefix remap (cluster-built indexes store paths
    relative to the cluster layout, e.g. database/MP/... vs database/datafiles/MP/...)."""
    if not prefix_map:
        return path
    old, new = prefix_map.split(":", 1)
    return path.replace(old, new, 1) if path.startswith(old) else path


def structure_to_ordered_atoms(structure):
    """Builder-convention ASE Atoms: one atom per site, dominant species.

    Returns (atoms, symbols). Raises ValueError for fully-vacant sites.
    """
    from ase import Atoms

    symbols = []
    for idx, site in enumerate(structure):
        sym = _dominant_symbol(_get_site_species_info(site, idx))
        if sym in VACANCY_SYMBOLS:
            raise ValueError(f"site {idx} is a vacancy placeholder ({sym})")
        symbols.append(sym)
    atoms = Atoms(
        symbols=symbols,
        scaled_positions=structure.frac_coords,
        cell=structure.lattice.matrix,
        pbc=True,
    )
    return atoms, symbols


def check_alignment(symbols, graph_path):
    """Strict per-structure check: our Z sequence == the graph's node Z sequence."""
    from pymatgen.core.periodic_table import Element

    with open(graph_path) as f:
        graph = json.load(f)
    graph_z = [int(n["Z"]) for n in graph["nodes"]]
    ours_z = [Element(s).Z for s in symbols]
    if graph_z != ours_z:
        raise ValueError(
            f"atom-order mismatch vs {os.path.basename(graph_path)}: "
            f"graph has {len(graph_z)} nodes {graph_z[:6]}..., "
            f"embed pass produced {len(ours_z)} atoms {ours_z[:6]}...")


def load_calculator(model="medium", device="cuda", dtype="float32"):
    from mace.calculators import mace_mp

    return mace_mp(model=model, device=device, default_dtype=dtype)


def _resolve_cif(cif_id, cif_dirs):
    for d in cif_dirs:
        p = os.path.join(d, cif_id)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{cif_id} not found in any of {cif_dirs}")


def embed_index(index_path, cif_dir, out_dir, model="medium", device="cuda",
                only_label=None, limit=None, prefix_map="database/MP:database/datafiles/MP",
                progress_every=200):
    """Embed every (or every label-matching) row of a cgv4 index. Resumable:
    rows whose .npy already exists are skipped. ``cif_dir`` may be one
    directory or a list tried in order (multi-source indexes keep their CIFs
    in per-source dirs)."""
    import pandas as pd

    cif_dirs = [cif_dir] if isinstance(cif_dir, str) else list(cif_dir)

    df = pd.read_pickle(index_path) if index_path.endswith(".pickle") else pd.read_csv(index_path)
    if only_label is not None and "label" in df.columns:
        df = df[df["label"] == only_label]
    if limit:
        df = df.head(limit)
    os.makedirs(out_dir, exist_ok=True)
    failed_path = os.path.join(out_dir, "failed.txt")

    calc = load_calculator(model=model, device=device)

    done = skipped = failed = 0
    dim = None
    t0 = time.time()
    with open(failed_path, "a") as failed_log:
        for _, row in df.iterrows():
            out_path = os.path.join(out_dir, row["id"] + ".npy")
            if os.path.exists(out_path):
                skipped += 1
                continue
            try:
                cif_path = _resolve_cif(row["id"], cif_dirs)
                structure = _load_unit_cell_structure_from_cif(cif_path)
                atoms, symbols = structure_to_ordered_atoms(structure)
                check_alignment(symbols, remap_graph_path(row["graph_path"], prefix_map))
                desc = calc.get_descriptors(atoms, invariants_only=True)
                desc = np.asarray(desc, dtype=np.float32)
                if desc.ndim != 2 or desc.shape[0] != len(structure):
                    raise ValueError(f"unexpected descriptor shape {desc.shape} "
                                     f"for {len(structure)} atoms")
                dim = desc.shape[1]
                np.save(out_path, desc)
                done += 1
            except Exception as exc:  # log-and-continue: one bad CIF must not kill the pass
                failed += 1
                failed_log.write(f"{row['id']}\t{type(exc).__name__}: {exc}\n")
                failed_log.flush()
            if (done + failed) % progress_every == 0 and (done + failed) > 0:
                rate = (done + failed) / (time.time() - t0)
                print(f"  embedded {done} (failed {failed}, skipped {skipped}) "
                      f"- {rate:.1f} struct/s", flush=True)

    manifest = {
        "source_model": f"mace_mp_0_{model}",
        "invariants_only": True,
        "dim": dim,
        "dtype": "float32",
        "index_path": index_path,
        "n_embedded": done,
        "n_failed": failed,
        "n_skipped_existing": skipped,
        "alignment_check": "per-structure Z sequence vs graph nodes (strict)",
    }
    # Merge counts if a manifest already exists (resumed runs).
    man_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(man_path):
        with open(man_path) as f:
            old = json.load(f)
        manifest["n_embedded"] += old.get("n_embedded", 0)
        manifest["dim"] = manifest["dim"] or old.get("dim")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"embed-mace done: {done} new, {skipped} existing, {failed} failed -> {out_dir}")
    return manifest
