"""Site-projected (per-ELEMENT) phonon DOS for the Togo + MP-pheasy sources ->
graphs_v45_phdos["phdos_site"] -> phdos_pack_v45_site (user-directed 2026-09-09).

Conventions are IDENTICAL to the Cerqueira site pack (build_phonon_targets.py
phdos-site / plot_phonon_dispersion.dense_phdos): PHONON_SITE_BINS=64 centers
on 0-PHONON_W_MAX_THZ, Gaussian sigma PHDOS_SIGMA_THZ=0.15, mode weight of atom
i = sum_alpha |e_{i alpha}(q,nu)|^2 (dynamical-matrix eigenvectors, partitions
each mode exactly), kept = fraction of positive-frequency modes, on-grid total
normalized to 3*kept*nat, per element = mean over that element's atoms (so each
element curve ~ integrates to 3*kept per atom; the loader expands per node by
element). Mesh 20^3 with time-reversal reduction only (projections need
is_mesh_symmetry=False); q-points solved in bounded chunks with eigenvectors.
Sources for the site data: Togo zips (force constants; DFT) and MP pheasy
force constants (API: get_forceconstants_from_material_id; MLP-derived). MP
DFPT rows have no force constants via the API and get no site data (NaN, masked).

Subcommands (all resumable, per-material isolation):
  fetch-fc     pheasy phonon-doc metadata + force constants -> MP_PhononDOS/fc_pheasy/<mid>.npz
  site-togo    TogoPhononDB/zips -> TogoPhononDB/site_raw/<mid>.npz
  site-pheasy  fc_pheasy -> MP_PhononDOS/site_raw/<mid>.npz
  bake         write g["phdos_site"] + g["phonon_site_grid"] into graphs_v45_phdos (togo > pheasy)
  report       coverage + normalization/consistency stats
  python database/pipelines/phonon/build_phdos_site_corpus.py <cmd> [--workers N] [--limit N]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("RAYON_NUM_THREADS", "2")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import glob  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import lzma  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import warnings  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root from database/pipelines/<group>/
sys.path.insert(0, os.path.join(_ROOT, "models", "common"))
from data import PHONON_SITE_BINS, PHONON_W_MAX_THZ  # noqa: E402

MPD = os.path.join(_ROOT, "database", "datafiles", "MP_PhononDOS")
TOGO = os.path.join(_ROOT, "database", "datafiles", "TogoPhononDB")
INDEX = os.path.join(MPD, "PHDOS_index.pickle")
FC_DIR = os.path.join(MPD, "fc_pheasy")
SITE_MP = os.path.join(MPD, "site_raw")
SITE_TOGO = os.path.join(TOGO, "site_raw")
SIGMA = 0.15
MESH = [20, 20, 20]
CHUNK_BYTES = 2e8          # eigenvector batch budget per run_qpoints call
WORKER_DATA_GB = float(os.environ.get("SITE_WORKER_DATA_GB", "4.5"))
CENTERS = (np.arange(PHONON_SITE_BINS) + 0.5) * (PHONON_W_MAX_THZ / PHONON_SITE_BINS)
DW = PHONON_W_MAX_THZ / PHONON_SITE_BINS


# ----------------------------------------------------------------- core ----
def site_dos(ph):
    """Per-element site-projected DOS dict, kept fraction, nat — Cerqueira
    convention (see module docstring) on a 20^3 mesh via chunked run_qpoints."""
    from phonopy.phonon.mesh import Mesh
    grid = Mesh(ph.dynamical_matrix, MESH, is_mesh_symmetry=False, is_time_reversal=True,
                rotations=ph.primitive_symmetry.pointgroup_operations,
                primitive_symmetry=ph.primitive_symmetry, factor=ph.unit_conversion_factor)
    q, w = grid.qpoints, grid.weights.astype(float)
    nat = len(ph.primitive)
    nb = 3 * nat
    chunk = max(1, int(CHUNK_BYTES // (16 * nb * nb)))
    spec = np.zeros((nat, PHONON_SITE_BINS))
    n_all = n_pos = 0.0
    for i in range(0, len(q), chunk):
        ph.run_qpoints(q[i:i + chunk], with_eigenvectors=True)
        f = np.asarray(ph.qpoints.frequencies, dtype=np.float64)        # (nq, nb)
        V = np.asarray(ph.qpoints.eigenvectors)                          # (nq, nb, nb), columns = modes
        wt = (np.abs(V) ** 2).reshape(len(f), nat, 3, nb).sum(axis=2)   # (nq, nat, nb)
        wq = w[i:i + chunk]
        pos = f > 0
        n_all += wq.sum() * nb
        n_pos += (pos * wq[:, None]).sum()
        g = np.exp(-0.5 * ((CENTERS[None, None, :] - f[:, :, None]) / SIGMA) ** 2)
        g *= (pos * wq[:, None])[:, :, None]
        spec += np.einsum("qam,qmb->ab", wt, g)
    kept = n_pos / max(n_all, 1.0)
    spec /= SIGMA * np.sqrt(2 * np.pi)
    tot = spec.sum() * DW
    if tot > 0:
        spec *= (3.0 * kept * nat) / tot
    symbols = list(ph.primitive.symbols)
    site = {el: spec[[k for k, s in enumerate(symbols) if s == el]].mean(axis=0).astype(np.float32)
            for el in sorted(set(symbols))}
    return site, float(kept), nat


def _save_site(dest, site, kept, nat, source):
    els = sorted(site)
    np.savez_compressed(dest + ".tmp.npz", elements=np.array(els), site=np.stack([site[e] for e in els]),
                        kept=kept, nat=nat, source=source)
    os.replace(dest + ".tmp.npz", dest)


def _init_worker():
    import resource
    lim = int(WORKER_DATA_GB * 2**30)
    resource.setrlimit(resource.RLIMIT_DATA, (lim, lim))


# ----------------------------------------------------------------- togo ----
def togo_one(zpath):
    mp_id = os.path.basename(zpath)[:-4]
    dest = os.path.join(SITE_TOGO, f"{mp_id}.npz")
    if os.path.exists(dest):
        return "skip"
    try:
        import phonopy
        with zipfile.ZipFile(zpath) as z:
            name = next(n for n in z.namelist() if n.endswith("phonopy_params.yaml.xz"))
            raw = lzma.decompress(z.read(name))
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(raw)
            tmp = f.name
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ph = phonopy.load(tmp, log_level=0)
                site, kept, nat = site_dos(ph)
        finally:
            os.unlink(tmp)
        if kept <= 0 or not all(np.isfinite(v).all() for v in site.values()):
            return f"fail {mp_id}: degenerate (kept {kept:.3f})"
        _save_site(dest, site, kept, nat, "togo")
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"fail {mp_id}: {type(e).__name__}: {str(e)[:100]}"


# --------------------------------------------------------------- pheasy ----
def _index_ids():
    import pandas as pd
    return pd.read_pickle(INDEX)


def cmd_fetch_fc(workers=4, limit=None):
    """Phonon-doc metadata (structure/supercell/primitive matrices/Born) + force
    constants for every index row without a Togo zip (pheasy-sourced rows and
    the DFPT rows Togo does not cover — a material may carry a pheasy doc too)."""
    from concurrent.futures import ThreadPoolExecutor
    from mp_api.client import MPRester
    key = os.environ.get("MP_API_KEY")
    if not key:
        sys.exit("MP_API_KEY not set")
    os.makedirs(FC_DIR, exist_ok=True)
    idx = _index_ids()
    togo = {f[:-4] for f in os.listdir(os.path.join(TOGO, "dos_raw"))}
    need = [i for i in idx.id if i not in togo and not os.path.exists(os.path.join(FC_DIR, f"{i}.npz"))]
    if limit:
        need = need[:limit]
    print(f"fetch-fc: {len(need)} materials to fetch", flush=True)
    # The phonon doc's `identifier` is the CALCULATION AlphaID, not the material
    # id (mp-1 -> aaaehlty = MPID 1958890), so a bulk search cannot be mapped
    # back; metadata is fetched per material inside the worker (2 calls each).
    with MPRester(key) as m:
        def one(mid):
            try:
                docs = m.materials.phonon.search(
                    material_ids=[mid], fields=["structure", "supercell_matrix", "primitive_matrix",
                                                "phonon_method", "born", "epsilon_static"])
                docs = [d for d in docs if d.phonon_method == "pheasy"]
            except Exception as e:  # noqa: BLE001
                return f"fail {mid}: meta {str(e)[:80]}"
            if not docs:
                return f"nodoc {mid}"
            d = docs[0]
            try:
                # the FC endpoint intermittently raises "table ph_force_constants
                # already exists" under concurrent requests — retry with backoff
                import time as _t
                for attempt in range(4):
                    try:
                        fc = np.asarray(m.materials.phonon.get_forceconstants_from_material_id(mid, "pheasy"), dtype=np.float32)
                        break
                    except Exception as e:  # noqa: BLE001
                        if attempt == 3:
                            raise
                        _t.sleep(2.0 * (attempt + 1) + np.random.rand())
                st = d.structure
                born = np.asarray(d.born, dtype=np.float32) if getattr(d, "born", None) is not None else np.zeros((0, 3, 3), np.float32)
                eps = np.asarray(d.epsilon_static, dtype=np.float32) if getattr(d, "epsilon_static", None) is not None else np.zeros((0, 3), np.float32)
                pm = np.asarray(d.primitive_matrix, dtype=np.float64) if d.primitive_matrix is not None else np.full((3, 3), np.nan)
                dest = os.path.join(FC_DIR, f"{mid}.npz")
                np.savez_compressed(dest + ".tmp.npz", fc=fc, lattice=np.asarray(st.lattice.matrix), frac=np.asarray(st.frac_coords),
                                    symbols=np.array([s.specie.symbol for s in st]), supercell_matrix=np.asarray(d.supercell_matrix, dtype=np.float64),
                                    primitive_matrix=pm, born=born, eps=eps)
                os.replace(dest + ".tmp.npz", dest)
                return "ok"
            except Exception as e:  # noqa: BLE001
                return f"fail {mid}: {str(e)[:100]}"
        n = {"ok": 0, "fail": 0, "nodoc": 0}
        with ThreadPoolExecutor(workers) as ex:
            for k, res in enumerate(ex.map(one, need)):
                n[res.split()[0]] += 1
                if not res.startswith("ok") and n["fail"] + n["nodoc"] <= 30:
                    print("  " + res, flush=True)
                if (k + 1) % 500 == 0:
                    print(f"  fc {k + 1}/{len(need)} {n}", flush=True)
    print(f"fetch-fc done: {n}", flush=True)


def pheasy_one(fc_path):
    mid = os.path.basename(fc_path)[:-4]
    dest = os.path.join(SITE_MP, f"{mid}.npz")
    if os.path.exists(dest):
        return "skip"
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
        z = np.load(fc_path, allow_pickle=False)
        cell = PhonopyAtoms(symbols=[str(s) for s in z["symbols"]], cell=z["lattice"], scaled_positions=z["frac"])
        pm = z["primitive_matrix"]
        pm = None if np.isnan(pm).any() else pm
        with contextlib.redirect_stdout(io.StringIO()):
            ph = Phonopy(cell, supercell_matrix=z["supercell_matrix"], primitive_matrix=pm, log_level=0)
            fc = np.asarray(z["fc"], dtype=np.float64)
            if fc.shape[0] not in (len(ph.supercell), len(ph.primitive)):
                return f"fail {mid}: fc {fc.shape} vs supercell {len(ph.supercell)}"
            ph.force_constants = fc
            born, eps = z["born"], z["eps"]
            if born.shape[0] == len(ph.primitive) and eps.shape == (3, 3):
                ph.nac_params = {"born": born.astype(np.float64), "dielectric": eps.astype(np.float64), "factor": 14.399652}
            site, kept, nat = site_dos(ph)
        if kept <= 0 or not all(np.isfinite(v).all() for v in site.values()):
            return f"fail {mid}: degenerate (kept {kept:.3f})"
        _save_site(dest, site, kept, nat, "pheasy")
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"fail {mid}: {type(e).__name__}: {str(e)[:100]}"


def _run_pool(fn, items, workers, label):
    from multiprocessing import Pool
    n = {"ok": 0, "skip": 0, "fail": 0}
    print(f"{label}: {len(items)} items, {workers} workers, RLIMIT_DATA {WORKER_DATA_GB} GiB/worker", flush=True)
    with Pool(workers, initializer=_init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(fn, items, chunksize=2)):
            n["fail" if res.startswith("fail") else res] += 1
            if res.startswith("fail") and n["fail"] <= 100:
                print("  " + res, flush=True)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(items)} {n}", flush=True)
    print(f"{label} done: {n}", flush=True)


def cmd_site_togo(workers=4, limit=None):
    os.makedirs(SITE_TOGO, exist_ok=True)
    idx = _index_ids()
    have = {f[:-4] for f in os.listdir(os.path.join(TOGO, "dos_raw"))}
    zips = [os.path.join(TOGO, "zips", f"{i}.zip") for i in idx.id if i in have]   # index rows with a Togo zip (any source)
    zips = [z for z in zips if os.path.exists(z)]
    _run_pool(togo_one, zips[:limit] if limit else zips, workers, "site-togo")


def cmd_site_pheasy(workers=4, limit=None):
    os.makedirs(SITE_MP, exist_ok=True)
    fcs = sorted(glob.glob(os.path.join(FC_DIR, "*.npz")))
    _run_pool(pheasy_one, fcs[:limit] if limit else fcs, workers, "site-pheasy")


# ----------------------------------------------------------------- bake ----
def _site_for(mid):
    for d, src in ((SITE_TOGO, "togo"), (SITE_MP, "pheasy")):
        p = os.path.join(d, f"{mid}.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=False)
            return {str(e): z["site"][k] for k, e in enumerate(z["elements"])}, float(z["kept"]), src
    return None, None, None


def cmd_bake(dry_run=False, **_):
    from pymatgen.core.periodic_table import Element
    idx = _index_ids()
    n = {"togo": 0, "pheasy": 0, "none": 0, "elmismatch": 0}
    kept_all = []
    for _, row in idx.iterrows():
        site, kept, src = _site_for(row["id"])
        if site is None:
            n["none"] += 1
            continue
        gpath = row["graph_path"] if os.path.isabs(row["graph_path"]) else os.path.join(_ROOT, row["graph_path"])
        g = json.load(open(gpath))
        gel = {Element.from_Z(int(nd["Z"])).symbol for nd in g["nodes"]}
        if not gel <= set(site):
            n["elmismatch"] += 1
            if n["elmismatch"] <= 10:
                print(f"  element mismatch {row['id']}: graph {sorted(gel)} vs site {sorted(site)}", flush=True)
            continue
        if not dry_run:
            g["phdos_site"] = {el: [round(float(x), 8) for x in v] for el, v in site.items()}
            g["phonon_site_grid"] = [PHONON_SITE_BINS, PHONON_W_MAX_THZ]
            g["phdos_site_source"] = src
            tmp = gpath + ".tmp"
            json.dump(g, open(tmp, "w"))
            os.replace(tmp, gpath)
        n[src] += 1
        kept_all.append(kept)
        if (n["togo"] + n["pheasy"]) % 2000 == 0:
            print(f"  baked {n['togo'] + n['pheasy']}", flush=True)
    print(f"bake: {n} | kept median {np.median(kept_all):.4f} p5 {np.percentile(kept_all, 5):.4f}" if kept_all else f"bake: {n}", flush=True)


def cmd_report(**_):
    idx = _index_ids()
    rows = []
    for mid, src in zip(idx.id, idx.phdos_source):
        site, kept, ssrc = _site_for(mid)
        if site is None:
            continue
        integ = np.array([v.sum() * DW for v in site.values()])
        rows.append((src, ssrc, kept, integ.mean() / max(3.0 * kept, 1e-9)))
    if not rows:
        print("no site data yet")
        return
    import pandas as pd
    d = pd.DataFrame(rows, columns=["dos_source", "site_source", "kept", "integral_over_3kept"])
    print(d.groupby(["dos_source", "site_source"]).agg(n=("kept", "size"), kept_med=("kept", "median"),
                                                        ratio_med=("integral_over_3kept", "median"),
                                                        ratio_p5=("integral_over_3kept", lambda x: x.quantile(.05)),
                                                        ratio_p95=("integral_over_3kept", lambda x: x.quantile(.95))).to_string())
    print(f"coverage: {len(d)}/{len(idx)} index rows have site data")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch-fc", "site-togo", "site-pheasy", "bake", "report"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"fetch-fc": lambda: cmd_fetch_fc(a.workers, a.limit), "site-togo": lambda: cmd_site_togo(a.workers, a.limit),
     "site-pheasy": lambda: cmd_site_pheasy(a.workers, a.limit), "bake": lambda: cmd_bake(a.dry_run),
     "report": cmd_report}[a.cmd]()
