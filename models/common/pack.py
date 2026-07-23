"""MPNNPack.py — packed columnar dataset: pack once, read fast forever.

Why: at MPtrj scale (~1.6M frames) the lazy loader pays ~4 ms/sample every epoch
(GPFS small-file open + json.loads + sort/pad/stack). Packing runs that work ONCE
and stores the *ragged extraction* (MPNNData._extract_ragged) as flat per-field
binary files + a per-sample offset table; training then memory-maps the arrays and
only pads (MPNNData._assemble_sample) at read time (~100-200 us/sample).

Storing the RAGGED form (not padded tensors) means max_num_nbr /
max_num_poly_nbr / use_poly_edges / use_bond_angles / build_angle_bias are all
READ-time parameters — one pack serves every config variant. Only a change to
graph *content* (a builder rebuild) invalidates a pack.

Because both backends assemble through the same `_assemble_sample` over the same
`_extract_ragged` output, packed samples are bitwise-identical to lazy ones.

Pack directory layout:
    pack_header.json   version, per-array element counts, source index, dims
    meta.pickle        DataFrame: id, label, [mp_id], target columns, offsets
    <field>.bin        raw little-endian arrays (see _FIELDS)
"""
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data import (CIFDataV4, _extract_ragged, _assemble_sample, _assemble_targets,
                      _select_target_key, build_data_rows, subsample_frames_by_group,
                      rows_meanstd,
                      accumulate_slot_rbf, rich_node_features, RICH_NODE_FEA_LEN,
                      valence_node_features, VALENCE_NODE_FEA_LEN, CF_FEA_LEN,
                      NODE_FEA_LEN, NBR_FEA_LEN, POLY_FEA_LEN, ANGLE_FEA_LEN,
                      DIHEDRAL_FEA_LEN, DOS_N_ENERGY)

PACK_VERSION = 1

# field name -> (dtype, row width or None for 1-D); order = write order.
# frac_coords is atom-aligned (N rows/sample, like atom_fea) so it reuses
# atom_start/n_atoms — no new offset column. lattice (3x3/sample) rides in meta.
# Legacy packs without these keys load fine (see _maps / _ragged guards).
_FIELDS = {
    "atom_fea":  (np.float32, NODE_FEA_LEN),
    "frac_coords": (np.float32, 3),
    "dih_node":  (np.float32, DIHEDRAL_FEA_LEN),   # atom-aligned 4-body torsion summary
    "valence":   (np.float32, VALENCE_NODE_FEA_LEN),  # baked subshells (v4.3 graphs; zeros else)
    "cf":        (np.float32, CF_FEA_LEN),            # baked AOM crystal-field block (v4.3)
    "forces":    (np.float32, 3),    # atom-aligned multitask TARGET (NaN where absent)
    "magmom":    (np.float32, 1),    # atom-aligned multitask TARGET (NaN where absent)
    "bond_cnt":  (np.int32, None),
    "bond_nbr":  (np.int32, None),
    "bond_fea":  (np.float32, NBR_FEA_LEN),
    "bond_jimage": (np.int16, 3),    # per-edge PBC image (center->neighbor) for exact bond vectors
    "poly_cnt":  (np.int32, None),
    "poly_nbr":  (np.int32, None),
    "poly_fea":  (np.float32, POLY_FEA_LEN),
    "ang_cnt":   (np.int32, None),
    "ang_slots": (np.int16, 2),
    "ang_cos":   (np.float32, None),
    "ang_vcnt":  (np.int32, None),
    "ang_vslot": (np.int16, None),
    "ang_vcos":  (np.float32, None),
}
# Per-sample offset columns in meta: start of this sample's rows in each ragged
# stream (the *_cnt arrays are atom-aligned, so they share atom_start/n_atoms).
_OFFSET_COLS = ("atom_start", "n_atoms", "bond_start", "n_bond",
                "poly_start", "n_poly", "ang_start", "n_ang",
                "angv_start", "n_angv")


def _extract_one(graph_path):
    """Worker: read one graph JSON -> ragged dict (or error string)."""
    try:
        return _extract_ragged(CIFDataV4._read_graph(graph_path)), None
    except Exception as exc:  # noqa: BLE001 — recorded per sample, build continues
        return None, f"{type(exc).__name__}: {exc}"


def pack_dataset(index_path, out_dir, n_workers=None, limit=None, chunksize=16,
                 derive_mp_id=False):
    """Pack every graph referenced by an index pickle into ``out_dir``.

    Extraction parallelizes across a process pool; the writer appends to the
    field files sequentially in index order (offsets must be deterministic).
    Samples whose graph fails to read are skipped and logged to failed.txt.

    ``derive_mp_id`` (opt-in): when the index has no ``mp_id`` column, synthesize
    one from ``id`` (stripping a trailing ``.cif``) so the pack supports a
    material-level split. For single-structure-per-material datasets (relaxed MP,
    e.g. the DOS pack) where ``id`` *is* the material id — needed when such a pack
    joins a material-split masked union (rung 04). OFF by default so it never
    silently flips the split of a dataset that intends a frame split (the
    MP_Energy eform benchmark) — those keep groups=None.
    """
    import multiprocessing as mp

    df = pd.read_pickle(index_path)
    if limit:
        df = df.head(limit)
    os.makedirs(out_dir, exist_ok=True)

    files = {name: open(os.path.join(out_dir, name + ".bin"), "wb")
             for name in _FIELDS}
    totals = {name: 0 for name in _FIELDS}   # element rows written per field
    offsets = {col: [] for col in _OFFSET_COLS}
    kept_pos, failures, lattices, dih_any, stresses, doses = [], [], [], [], [], []
    phys_any = {"forces": False, "magmom": False, "stress": False, "dos": False}  # any finite -> has_X
    baked_any = {"valence": False, "cf": False}  # v4.3 baked node blocks present -> has_*
    jimage_any = [False]   # any nonzero bond_jimage -> graphs carry exact PBC images
    start = time.time()

    def _write(name, arr, dtype):
        files[name].write(np.ascontiguousarray(arr, dtype=dtype).tobytes())
        totals[name] += len(arr)

    n_workers = (os.cpu_count() or 1) if n_workers is None else max(1, int(n_workers))
    paths = df["graph_path"].tolist()

    def _consume(results):
        for pos, (ragged, err) in enumerate(results):
            if err is not None:
                failures.append(f"{df.iloc[pos]['id']}\t{err}\n")
                continue
            r = ragged
            offsets["atom_start"].append(totals["atom_fea"])
            offsets["n_atoms"].append(r["n_atoms"])
            offsets["bond_start"].append(totals["bond_fea"])
            offsets["n_bond"].append(len(r["bond_fea"]))
            offsets["poly_start"].append(totals["poly_fea"])
            offsets["n_poly"].append(len(r["poly_fea"]))
            offsets["ang_start"].append(totals["ang_cos"])
            offsets["n_ang"].append(len(r["ang_cos"]))
            offsets["angv_start"].append(totals["ang_vcos"])
            offsets["n_angv"].append(len(r["ang_vcos"]))
            for name, (dtype, _w) in _FIELDS.items():
                arr = r.get(name)
                if arr is None and name in ("valence", "cf"):
                    # legacy graph without baked blocks: zero-fill the uniform bin;
                    # the has_* header flags below restore None semantics at read.
                    arr = np.zeros((int(r["n_atoms"]), _FIELDS[name][1]), dtype=dtype)
                _write(name, arr, dtype)
            lattices.append(np.asarray(r["lattice"], dtype=np.float32).reshape(9))
            stresses.append(np.asarray(r["stress"], dtype=np.float32).reshape(9))
            doses.append(np.asarray(r["dos"], dtype=np.float32).reshape(DOS_N_ENERGY))
            dih_any.append(bool(np.any(r["dih_node"])))
            if not baked_any["valence"] and r.get("valence") is not None:
                baked_any["valence"] = True
            if not baked_any["cf"] and r.get("cf") is not None:
                baked_any["cf"] = True
            for _k in phys_any:
                if not phys_any[_k] and np.isfinite(r[_k]).any():
                    phys_any[_k] = True
            if not jimage_any[0] and np.any(r["bond_jimage"]):
                jimage_any[0] = True
            kept_pos.append(pos)
            done = len(kept_pos) + len(failures)
            if done % 5000 == 0:
                rate = done / max(time.time() - start, 1e-9)
                print(f"  packed {done}/{len(paths)} ({rate:.0f} samples/s, "
                      f"{len(failures)} failed)", flush=True)

    if n_workers == 1:
        _consume(map(_extract_one, paths))
    else:
        with mp.Pool(processes=n_workers) as pool:
            _consume(pool.imap(_extract_one, paths, chunksize=chunksize))

    for f in files.values():
        f.close()

    # meta: original index columns (minus graph_path — packed reads need no JSON)
    # for the kept samples, plus the per-sample offsets.
    meta = df.iloc[kept_pos].drop(columns=["graph_path"], errors="ignore"
                                  ).reset_index(drop=True)
    for col in _OFFSET_COLS:
        meta[col] = np.asarray(offsets[col], dtype=np.int64)
    # lattice (3x3, flattened to 9) rides as a meta column, aligned with kept_pos.
    lat_stack = np.stack(lattices) if lattices else np.zeros((0, 9), dtype=np.float32)
    if lattices:
        meta["lattice"] = list(lat_stack)
    # Per-sample physics meta columns (stress 3x3 -> 9, total DOS (DOS_N_ENERGY,)),
    # aligned with kept_pos. Persisted ONLY when at least one sample has a REAL value
    # (phys_any) — an all-NaN column is pure fill whose width fossilizes the pack-time
    # grid and spuriously trips the reader's width check when the grid later changes
    # (the packed_v4 256-bin dos trap). Rule applies to every physics meta column.
    if stresses and phys_any["stress"]:
        meta["stress"] = list(np.stack(stresses))
    if doses and phys_any["dos"]:
        meta["dos"] = list(np.stack(doses))
    # Opt-in material-split key for single-structure-per-material packs (the DOS
    # pack): id IS the material id, so mp_id = id minus a trailing .cif. Only when
    # the index didn't already carry mp_id (never clobber an explicit one).
    if derive_mp_id and "mp_id" not in meta.columns and "id" in meta.columns:
        meta["mp_id"] = meta["id"].astype(str).str.replace(r"\.cif$", "", regex=True)
    meta.to_pickle(os.path.join(out_dir, "meta.pickle"))

    header = {
        "version": PACK_VERSION,
        "n_samples": len(kept_pos),
        "totals": totals,
        "dims": {"node": NODE_FEA_LEN, "edge": NBR_FEA_LEN,
                 "poly": POLY_FEA_LEN, "angle_rbf": ANGLE_FEA_LEN},
        "has_angles": totals["ang_cos"] > 0,
        "has_positions": bool(np.abs(lat_stack).sum() > 0),
        "has_dihedrals": bool(any(dih_any)),
        "has_valence_baked": bool(baked_any["valence"]),
        "has_cf": bool(baked_any["cf"]),
        "has_forces": phys_any["forces"],
        "has_magmom": phys_any["magmom"],
        "has_stress": phys_any["stress"],
        "has_dos": phys_any["dos"],
        # any nonzero per-edge image -> graphs carry EXACT to_jimage (rebuilt). False on a
        # pack of un-rebuilt graphs (all (0,0,0)) -> exact PBC forces would be wrong; verify
        # this is true on packed_v4 before multitask training.
        "has_to_jimage": jimage_any[0],
        "source_index": os.path.abspath(index_path),
        "n_failed": len(failures),
    }
    with open(os.path.join(out_dir, "pack_header.json"), "w") as f:
        json.dump(header, f, indent=2)
    if failures:
        with open(os.path.join(out_dir, "failed.txt"), "a") as f:
            f.writelines(failures)

    size_gb = sum(os.path.getsize(os.path.join(out_dir, n + ".bin"))
                  for n in _FIELDS) / 1e9
    print(f"Pack done: {len(kept_pos)} samples ({len(failures)} failed) in "
          f"{time.time()-start:.0f}s -> {out_dir} ({size_gb:.1f} GB)")
    return out_dir


class PackedCIFDataV4(Dataset):
    """Drop-in replacement for CIFDataV4 over a packed directory.

    Same constructor knobs, same .data/.labels/.groups/.target_column surface,
    same sample tuples (bitwise-identical to the lazy path) — samplers, splits,
    normalizer, and training code never know the difference. graph_cache_size is
    accepted-and-ignored (there is nothing worth caching: reads are ~100s of us).
    """

    def __init__(self, pack_dir, max_num_nbr=14, max_num_poly_nbr=16,
                 graph_cache_size=0, random_seed=123, target_column=None,
                 use_bond_angles=False, use_poly_edges=True,
                 build_angle_bias=False, use_rich_node_features=False,
                 use_dihedrals=False, multitask=False, frame_subsample=1,
                 use_valence_features=False, use_cf_features=False,
                 n_energy=None, dos_per_atom=True):
        with open(os.path.join(pack_dir, "pack_header.json")) as f:
            self._header = json.load(f)
        if self._header["version"] != PACK_VERSION:
            raise ValueError(f"pack version {self._header['version']} != "
                             f"reader version {PACK_VERSION}; re-pack {pack_dir}")
        # Feature dims are baked into BOTH the writer and this reader's memmap
        # shapes; a dim change without a version bump (it happened once: edge
        # 8 -> 7) would otherwise memmap with the wrong row width and silently
        # shift every row — the model would train on garbage with no error.
        current_dims = {"node": NODE_FEA_LEN, "edge": NBR_FEA_LEN,
                        "poly": POLY_FEA_LEN, "angle_rbf": ANGLE_FEA_LEN}
        if self._header.get("dims") != current_dims:
            raise ValueError(
                f"pack {pack_dir} was written with feature dims "
                f"{self._header.get('dims')} but this code uses {current_dims} — "
                "re-pack the dataset (python main.py pack-dataset ...).")
        # A pack that dropped failed samples has FEWER rows than its source index,
        # so the seeded shuffle permutes differently: train/val/test membership is
        # NOT comparable between this pack and the index pickle. Internally the
        # pack is still self-consistent — warn loudly rather than refuse.
        if self._header.get("n_failed", 0) > 0:
            print(f"WARNING: pack {pack_dir} dropped {self._header['n_failed']} "
                  "failed samples at pack time. Splits from this pack are NOT "
                  "comparable to splits from the source index pickle (different "
                  "row count -> different shuffle). Train AND evaluate against "
                  "the same backend.")
        if (use_bond_angles or build_angle_bias) and not self._header["has_angles"]:
            raise ValueError(
                "angle features requested but this pack holds no angle triplets "
                "(packed from pre-angle graphs). Rebuild graphs, re-pack, or "
                "disable use_bond_angles/the set_transformer aggregation.")

        self.pack_dir = pack_dir
        self.is_packed = True
        self.max_num_nbr = max_num_nbr
        self.max_num_poly_nbr = max_num_poly_nbr
        self.use_bond_angles = use_bond_angles
        self.use_poly_edges = use_poly_edges
        self.build_angle_bias = build_angle_bias
        self.use_rich_node_features = use_rich_node_features
        self.use_valence_features = use_valence_features
        self.use_cf_features = use_cf_features
        if use_cf_features and not self._header.get("has_cf"):
            raise ValueError(f"use_cf_features=True but pack {pack_dir} carries no baked "
                             "cf block — repack from builder-v4.3 graphs.")
        self.use_dihedrals = use_dihedrals
        self.multitask = multitask
        # DOS target width served by this reader: packs store whatever grid was current
        # at pack time (128-bin ±1 eV now; 256-bin -10..+5 eV in the legacy dos_pack), so
        # the configured n_energy selects which packs are compatible. NaN fills (DOS-less
        # packs in a union) use the same width so collate can stack across members.
        self.n_energy = int(n_energy) if n_energy else DOS_N_ENERGY
        self.dos_per_atom = dos_per_atom

        meta = pd.read_pickle(os.path.join(pack_dir, "meta.pickle"))
        # Multitask union members may lack the configured scalar target (the DOS
        # pack carries no formation energy) -> tolerate its absence and keep all
        # rows (the missing scalar is masked off; the pack's own targets train).
        self.target_column = target_key = _select_target_key(
            meta, target_column, pack_dir, required=not multitask)

        # Offset arrays in META ORDER; rows reference them by position so the
        # row shuffle below never reorders the arrays themselves.
        self._off = {c: meta[c].to_numpy() for c in _OFFSET_COLS}
        # Per-sample lattice (3x3), meta-order; None for legacy packs without it.
        self._lattice = (np.stack(meta["lattice"].to_numpy()).astype(np.float32).reshape(-1, 3, 3)
                         if "lattice" in meta.columns else None)
        # Multitask per-sample PHYSICS meta columns (meta-order, indexed by pos).
        # A column is loaded only when the header's has_X says the pack actually HAS
        # that target: legacy writers NaN-filled every sample (packed_v4/MPtrj holds a
        # 256-wide all-NaN dos column), and that fill is not data — width-validating
        # it against a newer grid spuriously rejected the pack (rungs 09/11), and
        # stacking it first cost ~1.5 GB of transient RSS per rank. The has_X check
        # comes BEFORE the stack; absent/dropped -> None -> NaN fill -> masked off.
        # has_X missing (pre-flag packs) -> trust the column, as before.
        def _phys_col(name, has_key):
            if name not in meta.columns or not self._header.get(has_key, True):
                return None
            return np.stack(meta[name].to_numpy()).astype(np.float32)

        _st = _phys_col("stress", "has_stress")
        self._stress = _st.reshape(-1, 3, 3) if _st is not None else None
        # DOS is stacked at its STORED width — a blind reshape(-1, n_energy) on a
        # mismatched pack would silently re-split rows (a legacy 256-bin pack read as
        # 128 doubles the row count). The width contract, enforced UNCONDITIONALLY
        # (not just multitask — _ragged serves r["dos"] to any reader):
        #   n_energy given  -> stored width must match, else raise;
        #   n_energy absent -> ADOPT the stored width (the pack is self-describing),
        #                      so casual readers get the true grid, never a mis-shape.
        self._dos = _phys_col("dos", "has_dos")
        if self._dos is not None:
            stored = int(self._dos.shape[1])
            if n_energy and stored != int(n_energy):
                raise ValueError(
                    f"pack {pack_dir} stores {stored}-bin DOS but the config requests "
                    f"n_energy={int(n_energy)} — point index_path at the matching pack "
                    f"(128 = ±1 eV dos_pack_ef1, 256 = legacy -10..+5 eV dos_pack) or "
                    f"fix n_energy.")
            self.n_energy = stored
        # Coerce non-numeric cells (e.g. '' for missing) to NaN before float cast.
        self._bandgap = (pd.to_numeric(meta["bandgap"], errors="coerce")
                         .to_numpy().astype(np.float32)
                         if "bandgap" in meta.columns else None)

        # Shared row construction (MPNNData.build_data_rows — bit-identical to
        # CIFDataV4 so the same seed yields the same splits across backends);
        # element 2 of each data tuple is the meta row position here.
        self.data, self.groups, self.labels, dropped = build_data_rows(
            meta, target_key, list(range(len(meta))), random_seed, keep_all=multitask)
        if dropped:
            print(f"PackedCIFDataV4: dropped {dropped} rows with no '{target_key}' value")
        if frame_subsample and frame_subsample > 1:
            n0 = len(self.data)
            self.data, self.groups, self.labels = subsample_frames_by_group(
                self.data, self.groups, self.labels, frame_subsample)
            print(f"PackedCIFDataV4: frame_subsample={frame_subsample} kept "
                  f"{len(self.data)}/{n0} frames (1-in-{frame_subsample} per material)")

        self._mm = None
        self._mm_pid = None

    # ------------------------------------------------------------------ mmaps
    def _maps(self):
        """Per-process lazily-opened memmaps (reopened after a worker fork)."""
        if self._mm is None or self._mm_pid != os.getpid():
            mm = {}
            for name, (dtype, width) in _FIELDS.items():
                n = self._header["totals"].get(name)
                if n is None:        # field absent in this (older) pack -> skip
                    continue
                shape = (n,) if width is None else (n, width)
                if n == 0:
                    # np.memmap raises on a zero-byte file; an empty field (e.g.
                    # a pack of angle-less graphs) just becomes an empty array.
                    mm[name] = np.zeros(shape, dtype=dtype)
                else:
                    mm[name] = np.memmap(os.path.join(self.pack_dir, name + ".bin"),
                                         dtype=dtype, mode="r", shape=shape)
            self._mm = mm
            self._mm_pid = os.getpid()
        return self._mm

    def _ragged(self, pos):
        """Ragged dict (memmap views) for meta row ``pos`` — _assemble_sample input."""
        mm = self._maps()
        o = self._off
        a0, n = int(o["atom_start"][pos]), int(o["n_atoms"][pos])
        b0, nb = int(o["bond_start"][pos]), int(o["n_bond"][pos])
        p0, npo = int(o["poly_start"][pos]), int(o["n_poly"][pos])
        g0, ng = int(o["ang_start"][pos]), int(o["n_ang"][pos])
        v0, nv = int(o["angv_start"][pos]), int(o["n_angv"][pos])
        frac = (mm["frac_coords"][a0:a0 + n] if "frac_coords" in mm
                else np.zeros((n, 3), dtype=np.float32))
        dih = (mm["dih_node"][a0:a0 + n] if "dih_node" in mm
               else np.zeros((n, DIHEDRAL_FEA_LEN), dtype=np.float32))
        # Baked v4.3 blocks: None (not zeros) when the pack predates them or was
        # packed from legacy graphs — assemble then falls back (valence) / raises
        # (cf), exactly matching the JSON backend's semantics.
        valence = (mm["valence"][a0:a0 + n]
                   if "valence" in mm and self._header.get("has_valence_baked") else None)
        cf = (mm["cf"][a0:a0 + n]
              if "cf" in mm and self._header.get("has_cf") else None)
        bjimage = (mm["bond_jimage"][b0:b0 + nb] if "bond_jimage" in mm
                   else np.zeros((nb, 3), dtype=np.int16))
        # Multitask targets: atom-aligned forces/magmom + per-sample stress. Absent
        # (legacy pack / no label) -> NaN so the masked loss skips them.
        forces = (mm["forces"][a0:a0 + n] if "forces" in mm
                  else np.full((n, 3), np.nan, dtype=np.float32))
        magmom = (mm["magmom"][a0:a0 + n] if "magmom" in mm
                  else np.full((n, 1), np.nan, dtype=np.float32))
        stress = (self._stress[pos] if self._stress is not None
                  else np.full((3, 3), np.nan, dtype=np.float32))
        dos = (self._dos[pos] if self._dos is not None
               else np.full(self.n_energy, np.nan, dtype=np.float32))
        lattice = (self._lattice[pos] if self._lattice is not None
                   else np.zeros((3, 3), dtype=np.float32))
        return {
            "n_atoms": n,
            "atom_fea": mm["atom_fea"][a0:a0 + n],
            "frac_coords": frac,
            "dih_node": dih,
            "valence": valence,
            "cf": cf,
            "lattice": lattice,
            "forces": forces,
            "magmom": magmom,
            "stress": stress,
            "dos": dos,
            "bond_cnt": mm["bond_cnt"][a0:a0 + n],
            "bond_nbr": mm["bond_nbr"][b0:b0 + nb],
            "bond_fea": mm["bond_fea"][b0:b0 + nb],
            "bond_jimage": bjimage,
            "poly_cnt": mm["poly_cnt"][a0:a0 + n],
            "poly_nbr": mm["poly_nbr"][p0:p0 + npo],
            "poly_fea": mm["poly_fea"][p0:p0 + npo],
            "ang_cnt": mm["ang_cnt"][a0:a0 + n],
            "ang_slots": mm["ang_slots"][g0:g0 + ng],
            "ang_cos": mm["ang_cos"][g0:g0 + ng],
            "ang_vcnt": mm["ang_vcnt"][a0:a0 + n],
            "ang_vslot": mm["ang_vslot"][v0:v0 + nv],
            "ang_vcos": mm["ang_vcos"][v0:v0 + nv],
        }

    # ----------------------------------------------------------------- Dataset
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cif_id, target, pos, label = self.data[idx]
        r = self._ragged(pos)
        sample = _assemble_sample(r, self.max_num_nbr, self.max_num_poly_nbr,
                                  self.use_poly_edges, self.use_bond_angles,
                                  self.build_angle_bias,
                                  use_rich_node_features=self.use_rich_node_features,
                                  use_dihedrals=self.use_dihedrals,
                                  use_valence_features=self.use_valence_features,
                                  use_cf_features=self.use_cf_features)
        if self.multitask:
            bg = float(self._bandgap[pos]) if self._bandgap is not None else float("nan")
            targets, masks = _assemble_targets(r, energy=target, bandgap=bg,
                                               dos_per_atom=self.dos_per_atom)
            return (sample, targets, masks, cif_id)
        return (sample, torch.FloatTensor([float(target)]),
                torch.LongTensor([int(label)]), cif_id)

    # -------------------------------------------------------------- interop
    def prebuild(self, device="cpu", progress_every=None):
        """No-op: packed reads are already ~100s of us; materializing 100s of GB
        of padded tensors is exactly what the pack exists to avoid."""
        print(">> packed dataset: prebuild skipped (unnecessary — reads are memmap-backed)")
        return self

    def feature_stats(self, indices, max_graphs=4000, seed=123):
        """Per-feature mean/std from the columnar arrays; mirrors
        compute_feature_stats semantics (full neighbor lists, no truncation;
        angle vectors per slot when use_bond_angles). Poly rows appear once per
        endpoint here vs once per edge lazily — mean/std are invariant to that
        uniform duplication (differences are at fp-rounding level)."""
        idx = list(indices)
        if max_graphs and len(idx) > max_graphs:
            idx = random.Random(seed).sample(idx, max_graphs)

        edge_dim = NBR_FEA_LEN + (ANGLE_FEA_LEN if self.use_bond_angles else 0)
        node_rows, edge_rows, poly_rows = [], [], []
        for i in idx:
            r = self._ragged(self.data[i][2])
            node = np.asarray(r["atom_fea"], dtype=np.float32)
            if self.use_rich_node_features:   # match assemble order: [base | rich | val | dih]
                node = np.concatenate([node, rich_node_features(node[:, 0])], axis=1)
            if self.use_valence_features:
                val = (np.asarray(r["valence"], dtype=np.float32)
                       if r.get("valence") is not None
                       else valence_node_features(node[:, 0], node[:, 1]))
                node = np.concatenate([node, val], axis=1)
            if self.use_cf_features:
                node = np.concatenate([node, np.asarray(r["cf"], dtype=np.float32)], axis=1)
            if self.use_dihedrals:
                node = np.concatenate([node, np.asarray(r["dih_node"], dtype=np.float32)], axis=1)
            node_rows.append(node)
            bond = np.asarray(r["bond_fea"], dtype=np.float32)
            if self.use_bond_angles:
                # Per-slot mean RBF via the shared accumulator (same math as
                # sample assembly, so the stats can't drift from the features).
                ang = np.zeros((len(bond), ANGLE_FEA_LEN), dtype=np.float32)
                cnt = np.zeros(len(bond), dtype=np.int64)
                b0 = v0 = 0
                for a in range(int(r["n_atoms"])):
                    c = int(r["bond_cnt"][a])
                    v = int(r["ang_vcnt"][a])
                    if v:
                        a_acc, a_cnt = accumulate_slot_rbf(
                            r["ang_vslot"][v0:v0 + v], r["ang_vcos"][v0:v0 + v], c)
                        ang[b0:b0 + c] += a_acc
                        cnt[b0:b0 + c] += a_cnt
                    b0 += c
                    v0 += v
                nz = cnt > 0
                ang[nz] /= cnt[nz, None]
                bond = np.concatenate([bond, ang], axis=1)
            edge_rows.append(bond)
            poly_rows.append(np.asarray(r["poly_fea"], dtype=np.float32))

        def _cat(rows, width):
            rows = [x for x in rows if len(x)]
            return np.concatenate(rows, axis=0) if rows else np.zeros((0, width), np.float32)

        node_dim = (NODE_FEA_LEN + (RICH_NODE_FEA_LEN if self.use_rich_node_features else 0)
                    + (VALENCE_NODE_FEA_LEN if self.use_valence_features else 0)
                    + (CF_FEA_LEN if self.use_cf_features else 0)
                    + (DIHEDRAL_FEA_LEN if self.use_dihedrals else 0))
        return {
            "node": rows_meanstd(_cat(node_rows, node_dim), node_dim),
            "edge": rows_meanstd(_cat(edge_rows, edge_dim), edge_dim),
            "poly": rows_meanstd(_cat(poly_rows, POLY_FEA_LEN), POLY_FEA_LEN),
        }
