"""Multi-source phonon-DOS corpus -> cgv4 graphs + baked ph_dos + pack (tasks #8/#9).

Sources, per mp-id, priority dfpt > togo > pheasy (curation quality ordering):
  dfpt   — MP DFPT (Petretto ~1.5k): MP_PhononDOS/raw/<id>.npz (method='dfpt'),
           structure from MP (fetch-structures cache).
  togo   — Togo phonondb (PBEsol phonopy, ~10k): TogoPhononDB/dos_raw/<id>.npz,
           structure from the zip's phonopy_params.yaml primitive cell (the cell
           the phonons were computed on — NOT refetched from MP).
  pheasy — MP pheasy 2025 expansion (~26k): same npz store as dfpt.

Normalization (one convention for all sources, applied at bake time from the
RAW curves — the Togo processor's cached 'binned' force-normalized all positive
states into the window, which distorts materials with modes above 60 THz):
  graph value g["ph_dos"] is EXTENSIVE: in-window integral =
  3 * n_graph_atoms * frac_win, where frac_win = (states in 0<w<=60 THz) /
  (all states incl. imaginary). _assemble_targets divides by n_atoms, matching
  the Cerqueira eph_pack convention exactly.

Subcommands (each resumable):
  fetch-structures  — bulk MP structure fetch for dfpt/pheasy-sourced ids ->
                      MP_PhononDOS/structures.jsonl.gz (skips banked ids)
  build             — source table + graphs via generate_CGv4_DB_from_structures
                      -> MP_PhononDOS/graphs_v45_phdos + PHDOS_index(.pickle)
  bake              — write g["ph_dos"]/g["phonon_grid"] into each graph JSON;
                      cross-source agreement report on togo∩MP overlap ids
  Then: python main.py pack-dataset --index .../PHDOS_index.pickle \
            --out database/datafiles/MP_PhononDOS/phdos_pack_v45

  python database/pipelines/phonon/build_phdos_corpus.py fetch-structures|build|bake
"""
import argparse
import glob
import gzip
import json
import lzma
import os
import sys
import zipfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root from database/pipelines/<group>/
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "models", "common"))
from data import PHONON_N_BINS, PHONON_W_MAX_THZ, bin_spectrum  # noqa: E402

MPD = os.path.join(_ROOT, "database", "datafiles", "MP_PhononDOS")
TOGO = os.path.join(_ROOT, "database", "datafiles", "TogoPhononDB")
STRUCTS = os.path.join(MPD, "structures.jsonl.gz")
GRAPHS = os.path.join(MPD, "graphs_v45_phdos")
INDEX = os.path.join(MPD, "PHDOS_index")
DW = PHONON_W_MAX_THZ / PHONON_N_BINS


def source_table():
    """{mp_id: source} with priority dfpt > togo > pheasy. Reads every MP npz's
    method field (~1 min for 26k) + the togo dos_raw listing."""
    togo = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(TOGO, "dos_raw", "*.npz"))}
    src = {}
    for p in glob.glob(os.path.join(MPD, "raw", "*.npz")):
        mid = os.path.basename(p)[:-4]
        try:
            with np.load(p) as z:
                src[mid] = str(z["method"])
        except Exception as e:  # noqa: BLE001 — corrupt npz shouldn't kill the build
            print(f"  bad npz {mid}: {str(e)[:60]}", flush=True)
    n_mp = len(src)
    for mid in togo:
        if src.get(mid) != "dfpt":
            src[mid] = "togo"
    counts = {s: sum(1 for v in src.values() if v == s) for s in ("dfpt", "togo", "pheasy")}
    print(f"source table: {len(src)} ids ({counts}); mp raw {n_mp}, togo dos {len(togo)}")
    return src


def togo_structure(mp_id):
    """Primitive cell from the zip's phonopy_params.yaml.xz, parsing only the
    header (the displacements/force_constants bulk is cut before yaml.load)."""
    import yaml
    from pymatgen.core import Lattice, Structure
    with zipfile.ZipFile(os.path.join(TOGO, "zips", f"{mp_id}.zip")) as z:
        name = next(n for n in z.namelist() if n.endswith("phonopy_params.yaml.xz"))
        txt = lzma.decompress(z.read(name)).decode()
    for cut in ("\ndisplacements:", "\nforce_constants:", "\nsupercell:"):
        i = txt.find(cut)
        if i >= 0:
            txt = txt[:i]
    doc = yaml.safe_load(txt)
    cell = doc.get("primitive_cell") or doc.get("unit_cell")
    if not cell:
        raise ValueError("no primitive_cell/unit_cell block")
    return Structure(Lattice(cell["lattice"]),
                     [p["symbol"] for p in cell["points"]],
                     [p["coordinates"] for p in cell["points"]]).as_dict()


def banked_structures():
    ids = set()
    if os.path.exists(STRUCTS):
        with gzip.open(STRUCTS, "rt") as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    continue
    return ids


def cmd_fetch_structures(limit=None):
    from mp_api.client import MPRester
    key = os.environ.get("MP_API_KEY")
    if not key:
        sys.exit("MP_API_KEY not set")
    src = source_table()
    need = sorted(m for m, s in src.items() if s in ("dfpt", "pheasy"))
    have = banked_structures()
    need = [m for m in need if m not in have]
    if limit:
        need = need[:limit]
    print(f"{len(need)} structures to fetch ({len(have)} banked)", flush=True)
    n_ok = n_fail = 0
    with MPRester(key) as m:
        for i in range(0, len(need), 250):
            chunk = need[i:i + 250]
            try:
                docs = m.materials.summary.search(
                    material_ids=chunk, fields=["material_id", "structure"])
            except Exception as e:  # noqa: BLE001 — log and move to the next chunk
                print(f"  chunk {i}: fetch failed {str(e)[:80]}", flush=True)
                n_fail += len(chunk)
                continue
            got = {str(d.material_id): d.structure for d in docs}
            with gzip.open(STRUCTS, "at") as f:
                for mid in chunk:
                    st = got.get(mid)
                    if st is None:
                        n_fail += 1
                        continue
                    f.write(json.dumps({"id": mid, "structure": st.as_dict()}) + "\n")
                    n_ok += 1
            if (i // 250) % 8 == 0:
                print(f"  {i + len(chunk)}/{len(need)} (ok {n_ok}, missing {n_fail})",
                      flush=True)
    print(f"fetch-structures done: ok {n_ok}, missing {n_fail}", flush=True)


def cmd_build():
    from database.database_main import generate_CGv4_DB_from_structures
    src = source_table()

    def records():
        n_mp = n_togo = n_skip = 0
        if os.path.exists(STRUCTS):
            with gzip.open(STRUCTS, "rt") as f:
                for line in f:
                    row = json.loads(line)
                    s = src.get(row["id"])
                    if s in ("dfpt", "pheasy"):
                        n_mp += 1
                        yield {"id": row["id"], "structure": row["structure"],
                               "mp_id": row["id"]}
        for mid, s in sorted(src.items()):
            if s != "togo":
                continue
            try:
                st = togo_structure(mid)
            except Exception as e:  # noqa: BLE001 — per-material isolation
                n_skip += 1
                if n_skip <= 15:
                    print(f"  togo structure fail {mid}: {str(e)[:80]}", flush=True)
                continue
            n_togo += 1
            yield {"id": mid, "structure": st, "mp_id": mid}
        print(f"records: {n_mp} mp + {n_togo} togo ({n_skip} togo structure fails)",
              flush=True)

    generate_CGv4_DB_from_structures(records(), output_dir=GRAPHS,
                                     output_index=INDEX, label=1)
    # stamp the source column onto the index for per-source weighting later
    import pandas as pd
    idx = pd.read_pickle(INDEX + ".pickle")
    idx["phdos_source"] = idx["id"].map(src)
    idx.to_pickle(INDEX + ".pickle")
    print(f"index: {len(idx)} rows, sources "
          f"{idx['phdos_source'].value_counts().to_dict()}")


def _per_atom_binned(w, d, n_atoms=None):
    """(per_atom_binned float32[NBINS], frac_win, n_atoms_inferred) from a raw
    DOS curve. States counted on the RAW curve; in-window (0, W_MAX] weight is
    what the binned vector integrates to (out-of-window weight stays absent)."""
    w = np.asarray(w, dtype=np.float64)
    d = np.clip(np.asarray(d, dtype=np.float64), 0.0, None)
    order = np.argsort(w)
    w, d = w[order], d[order]
    total = np.trapezoid(d, w)
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"degenerate DOS (total {total:.3g})")
    if n_atoms is None:
        n_atoms = max(int(round(total / 3.0)), 1)
    pos = w > 0
    if not pos.any():
        raise ValueError("no positive-frequency weight")
    win = pos & (w <= PHONON_W_MAX_THZ)
    frac_win = np.trapezoid(d[win], w[win]) / total if win.any() else 0.0
    if frac_win <= 0:
        raise ValueError("no in-window weight")
    binned = bin_spectrum(w[pos], d[pos]).astype(np.float64)
    s = binned.sum() * DW
    if s <= 0:
        raise ValueError("empty binned spectrum")
    target = 3.0 * frac_win  # per atom
    return (binned * (target / (s / n_atoms)) / n_atoms).astype(np.float32), frac_win, n_atoms


def load_per_atom(mid, source):
    """Per-atom binned ph_dos for one id from its selected source."""
    if source == "togo":
        with np.load(os.path.join(TOGO, "dos_raw", f"{mid}.npz")) as z:
            return _per_atom_binned(z["frequencies"], z["densities"],
                                    n_atoms=int(z["n_prim"]))
    with np.load(os.path.join(MPD, "raw", f"{mid}.npz")) as z:
        return _per_atom_binned(z["frequencies"], z["densities"])


def cmd_bake(dry_run=False):
    import pandas as pd
    idx = pd.read_pickle(INDEX + ".pickle")
    src = dict(zip(idx["id"], idx["phdos_source"]))
    n_ok = n_fail = 0
    fracs = []
    for _, row in idx.iterrows():
        gpath = os.path.join(_ROOT, row["graph_path"]) \
            if not os.path.isabs(row["graph_path"]) else row["graph_path"]
        if not os.path.exists(gpath):
            gpath = os.path.join(GRAPHS, f"{row['id']}.json")
        try:
            per_atom, frac_win, _ = load_per_atom(row["id"], src[row["id"]])
        except Exception as e:  # noqa: BLE001 — per-material isolation
            n_fail += 1
            if n_fail <= 15:
                print(f"  skip {row['id']}: {str(e)[:80]}", flush=True)
            continue
        if not dry_run:
            g = json.load(open(gpath))
            already = g.get("ph_dos")
            n_graph = len(g["nodes"])
            g["ph_dos"] = [round(float(x) * n_graph, 8) for x in per_atom]
            g["phonon_grid"] = [PHONON_N_BINS, PHONON_W_MAX_THZ]
            if already is None or len(already) != PHONON_N_BINS or \
                    abs(sum(already) - sum(g["ph_dos"])) > 1e-6:
                tmp = gpath + ".tmp"
                json.dump(g, open(tmp, "w"))
                os.replace(tmp, gpath)
        fracs.append(frac_win)
        n_ok += 1
        if n_ok % 2000 == 0:
            print(f"  baked {n_ok}", flush=True)
    fracs = np.array(fracs)
    print(f"bake: {n_ok} baked, {n_fail} failed")
    if len(fracs):
        print(f"in-window state fraction: median {np.median(fracs):.4f} "
              f"p5 {np.percentile(fracs, 5):.4f} | <90%: {(fracs < 0.9).sum()}")
    cross_source_report(src)


def cross_source_report(src, n_max=400):
    """Agreement on ids carried by BOTH togo and an MP method: per-atom binned
    curves' cosine similarity + integral ratio (togo vs MP)."""
    togo = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(TOGO, "dos_raw", "*.npz"))}
    both = sorted(m for m in src if m in togo
                  and os.path.exists(os.path.join(MPD, "raw", f"{m}.npz")))[:n_max]
    cos, ratio = [], []
    for mid in both:
        try:
            a, _, _ = load_per_atom(mid, "togo")
            with np.load(os.path.join(MPD, "raw", f"{mid}.npz")) as z:
                b, _, _ = _per_atom_binned(z["frequencies"], z["densities"])
        except Exception:  # noqa: BLE001
            continue
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            cos.append(float(a @ b / (na * nb)))
            ratio.append(float(a.sum() / max(b.sum(), 1e-9)))
    if cos:
        cos, ratio = np.array(cos), np.array(ratio)
        print(f"cross-source (togo vs MP, n={len(cos)}): cosine median "
              f"{np.median(cos):.3f} p10 {np.percentile(cos, 10):.3f} | "
              f"integral ratio median {np.median(ratio):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch-structures", "build", "bake"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "fetch-structures":
        cmd_fetch_structures(limit=a.limit)
    elif a.cmd == "build":
        cmd_build()
    else:
        cmd_bake(dry_run=a.dry_run)
