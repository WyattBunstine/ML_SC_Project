import os
import time
import warnings
import threading

import pandas as pd
import pymatgen
import pymatgen.core.structure       # ensure pymatgen.core.* submodules are bound
import pymatgen.core.periodic_table
import numpy as np
import json


def customwarn(message, category, filename, lineno, file=None, line=None):
    1+1 # sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))


# Recognized regression-target columns a source CSV may carry. The cgv4 index
# preserves every one that is present so the MPNN can pick which to train on via
# the config's `target_column` key (see models/MPNN/MPNNData.py). `tc` stays the
# default/legacy target; extend this tuple to add new targets.
KNOWN_TARGET_COLUMNS = ("tc", "e_above_hull", "formation_energy_per_atom", "energy_per_atom")


def _load_id_prop(csv_path, has_header=True):
    """Read an id->property CSV into a DataFrame with a 'cif' column plus every
    recognized target column present (see ``KNOWN_TARGET_COLUMNS``).

    has_header=True  : the CSV has a header row containing (at least) a 'cif'
                       column and one or more target columns, e.g. 3DSC_MP.csv
                       ('tc') or mp_energy.csv ('e_above_hull',
                       'formation_energy_per_atom'). Unrecognized columns dropped.
    has_header=False : the CSV has no header and two columns ordered
                       (cif_filename, tc), e.g. id_prop.csv.
    """
    if has_header:
        df = pd.read_csv(csv_path)
        keep = ["cif"] + [c for c in KNOWN_TARGET_COLUMNS if c in df.columns]
        return df.drop(df.columns.difference(keep), axis=1)
    return pd.read_csv(csv_path, header=None, names=["cif", "tc"])


def _unpack_source(data_file):
    """Unpack a source spec into (csv_path, cif_dir, label).

    A source spec is ``[csv_path, cif_dir]`` or ``[csv_path, cif_dir, label]``.
    ``label`` is the SC/non-SC class written into the database's ``label`` column:
    1 = superconductor (default), 0 = non-superconductor. It is set per source,
    NOT derived from the T_c value (note ~31% of the SC dataset has T_c=0.0, so a
    value-derived label would be wrong).
    """
    csv_path, cif_dir = data_file[0], data_file[1]
    label = int(data_file[2]) if len(data_file) > 2 else 1
    return csv_path, cif_dir, label


def generate_atom_init(output_file='database/datafiles/atom_init.json', max_z=95):
    """Build the per-element feature file consumed by the CNN's AtomInitializer.

    Writes a JSON object mapping each atomic number Z (1 .. max_z - 1) to a feature
    vector: [Z, block, valence, atomic_radius, electron_affinity, ionization_energy,
    electronegativity, electron_affinity].

    Default max_z=95 covers Z 1..94 (through Pu): the Materials Project formation-
    energy data includes actinide-bearing compounds (U, Th, Pu, ...), and the ORIG
    loader asserts on any element without an embedding here. Bump max_z further if a
    dataset introduces still-heavier elements.
    """
    elements = {}
    for i in np.arange(1, max_z):
        el = pymatgen.core.periodic_table.Element.from_Z(int(i))
        ele = []
        ele.append(el.Z)
        if el.block == 's':
            ele.append(0)
        if el.block == 'p':
            ele.append(1)
        if el.block == 'd':
            ele.append(2)
        if el.block == 'f':
            ele.append(3)
        x = el.group
        val = x
        if 56 < el.Z < 71:
            val = el.Z - 56
        if 3 <= x <= 12:
            val = x - 3
        if x > 12:
            val = x - 13
        ele.append(val)
        ele.append(el.atomic_radius)
        ele.append(el.electron_affinity)
        ele.append(el.ionization_energy)
        ele.append(el.X)
        ele.append(el.electron_affinity)
        elements[int(i)] = ele

    with open(output_file, 'w') as f:
        json.dump(elements, f)


def Proc_Basic_Batch(df, out_rows, cif_loc, thread_num, label=1):
    """Parse each CIF in ``df`` and append a dict row to ``out_rows`` carrying
    struc_dict, label, the legacy ``value`` (mirrors 'tc' when present), and every
    recognized target column the source provides (tc / e_above_hull /
    formation_energy_per_atom). The caller assembles the DataFrame once.

    Mirrors generate_CGv4_DB so the basic/CGCNN path supports the SAME multi-target
    selection: all target columns are written to the pickle and the consumer
    (CIFData) picks one at train time via `target_column`. Appending to a list is
    O(1); the DataFrame is built once by the caller. (The previous
    ``DataFrame.loc[...] = row`` enlargement was O(n^2) and became pathologically
    slow on the ~61k-row combined dataset.)"""
    warnings.showwarning = customwarn
    target_cols = [c for c in KNOWN_TARGET_COLUMNS if c in df.columns]
    if not target_cols:
        raise ValueError(
            "Basic DB build requires at least one recognized target column "
            f"({', '.join(KNOWN_TARGET_COLUMNS)}), but the source CSV has none "
            f"(columns: {list(df.columns)}).")
    t1 = time.time()
    total = len(df)
    for j, (index, row) in enumerate(df.iterrows(), 1):
        structure = pymatgen.core.structure.Structure.from_file(cif_loc + row['cif'])
        rec = {"id": row['cif'], "struc_dict": structure.as_dict(), "label": label,
               # Legacy single-target column: mirrors 'tc' only, left empty for
               # tc-less sources (e.g. the energy dataset) so a consumer that
               # forgets to set `target_column` errors instead of training on the
               # wrong target. Same convention as generate_CGv4_DB.
               "value": row['tc'] if 'tc' in target_cols else None}
        for c in target_cols:
            rec[c] = row[c]
        out_rows.append(rec)
        if j % 500 == 0:
            print("  thread " + str(thread_num) + ": " + str(j) + "/" + str(total) +
                  " (" + str(round(100 * j / total, 1)) + "%)  " +
                  str(round(time.time() - t1, 1)) + "s")
    print("Thread " + str(thread_num) + " time: " + str(round(time.time()-t1, 4)) + " seconds.")


def _compact_v4_graph(graph: dict) -> dict:
    """Reduce a full crystal_graph_v4 dict to the compact feature-only form the
    MPNN consumes (nodes, bonding edges + adjacency, polyhedral edges +
    adjacency). Pure function — safe to call from worker processes."""
    from pymatgen.core.periodic_table import Element as PmgElement

    ion_role_map = {"cation": 1, "anion": -1, "neutral": 0}
    compact_nodes = []
    for node in graph["nodes"]:
        el = PmgElement(node["element"])
        z = el.Z
        # Free-atom electronic props (eV), pure per-element lookups carried over
        # from the original CGCNN atom_init vector. None for elements without
        # tabulated data -> 0.0 so they don't poison the feature vector.
        ie = el.ionization_energy
        ea = el.electron_affinity
        hist = node.get("sharing_mode_hist") or {"corner": 0, "edge": 0, "face": 0, "other": 0}
        compact_nodes.append({
            "Z": z,
            "oxidation_state": float(node.get("oxidation_state") or 0.0),
            "ion_role": ion_role_map.get(node.get("ion_role", "neutral"), 0),
            "chi_pauling": float(node["chi_pauling"]) if node.get("chi_pauling") is not None else 0.0,
            "chi_allen": float(node["chi_allen"]) if node.get("chi_allen") is not None else 0.0,
            "ecn_value": float(node.get("ecn_value") or 0.0),
            "shannon_radius": float(node.get("shannon_radius_angstrom") or 0.0),
            "cn_core": int(node.get("cn_core") or 0),
            "hist_corner": int(hist.get("corner", 0)),
            "hist_edge": int(hist.get("edge", 0)),
            "hist_face": int(hist.get("face", 0)),
            "hist_other": int(hist.get("other", 0)),
            "ionization_energy": float(ie) if ie is not None else 0.0,
            "electron_affinity": float(ea) if ea is not None else 0.0,
        })

    compact_edges = []
    for edge in graph["edges"]:
        compact_edges.append({
            "id": edge["id"],
            # Endpoint node ids, kept so the MPNN data layer can orient the
            # directional (src/tgt) edge features RELATIVE TO THE CENTER atom when
            # building each atom's neighbor list — otherwise voronoi_/ecn_weight
            # src/tgt are in fixed storage order and are wrong for the ~half of
            # directed edges whose center is the stored target. Legacy graphs that
            # predate these keys fall back to the stored order (see MPNNData).
            "source": int(edge["source"]),
            "target": int(edge["target"]),
            "bond_length": float(edge["bond_length"]),
            "bond_length_over_sum_radii": float(edge["bond_length_over_sum_radii"])
                if edge.get("bond_length_over_sum_radii") is not None else 0.0,
            "voronoi_weight_src": float(edge["voronoi_weight_source"]),
            "voronoi_weight_tgt": float(edge["voronoi_weight_target"]),
            "ecn_weight_src": float(edge["ecn_weight_source"]),
            "ecn_weight_tgt": float(edge["ecn_weight_target"]),
            "delta_chi_pauling": float(edge["delta_chi_pauling"])
                if edge.get("delta_chi_pauling") is not None else 0.0,
            "coord_sphere": 1 if edge.get("coordination_sphere") == "core" else 0,
        })

    # Adjacency: node_id -> [(edge_id, neighbor_id), ...].
    # v4 graphs don't expose a top-level "adjacency"; rebuild it from the
    # (undirected) edge list. Isolated nodes keep an empty list so every node
    # id is present for the model.
    compact_adj = {str(nid): [] for nid in range(len(graph["nodes"]))}
    for edge in graph["edges"]:
        s, t, eid = edge["source"], edge["target"], edge["id"]
        compact_adj[str(s)].append((eid, t))
        compact_adj[str(t)].append((eid, s))

    # Polyhedral edges — second-neighbour connections through a shared bridging
    # atom (corner/edge/face sharing). A distinct edge type from bonding edges;
    # stored as their own compact list + adjacency so the model can message-pass
    # over them separately. path_type is encoded numerically the same way
    # ion_role is on nodes.
    poly_type_map = {"cation-anion-cation": 1, "anion-cation-anion": -1, "other": 0}
    compact_poly_edges = []
    for pe in graph.get("polyhedral_edges", []):
        compact_poly_edges.append({
            "id": pe["id"],
            "shared_count": int(pe.get("shared_count") or 0),
            "mean_angle_deg": float(pe.get("mean_angle_deg") or 0.0),
            "std_angle_deg": float(pe.get("std_angle_deg") or 0.0),
            "mean_path_length": float(pe.get("mean_path_length") or 0.0),
            "std_path_length": float(pe.get("std_path_length") or 0.0),
            "direct_distance": float(pe.get("direct_distance") or 0.0),
            "path_type": poly_type_map.get(pe.get("path_type", "other"), 0),
        })

    # Poly adjacency: node_id -> [(poly_edge_id, neighbor_id), ...].
    # Built from polyhedral_edges' (node_a, node_b) endpoints, same undirected
    # convention as the bonding adjacency above.
    compact_poly_adj = {str(nid): [] for nid in range(len(graph["nodes"]))}
    for pe in graph.get("polyhedral_edges", []):
        a, b, pid = pe["node_a"], pe["node_b"], pe["id"]
        compact_poly_adj[str(a)].append((pid, b))
        compact_poly_adj[str(b)].append((pid, a))

    # Bond-angle triplets (3-body): one cos(theta) per pair of bonding edges at a
    # shared center atom. Stored compactly as [center, edge_a, edge_b, cos] so the
    # MPNN dataset can expand cos in an RBF basis and aggregate onto the two edges.
    # Older full graphs without this key compact to an empty list (feature off).
    compact_triplets = [
        [int(t["center"]), int(t["edge_a"]), int(t["edge_b"]), float(t["cos_angle"])]
        for t in graph.get("angle_triplets", [])
    ]

    # Dihedrals (4-body): [central_edge, edge_i, edge_l, cos_dihedral]. Stored for a
    # future 4-body experiment; not consumed by the model yet.
    compact_dihedrals = [
        [int(d["central_edge"]), int(d["edge_i"]), int(d["edge_l"]), float(d["cos_dihedral"])]
        for d in graph.get("dihedrals", [])
    ]

    return {
        "nodes": compact_nodes,
        "edges": compact_edges,
        "adjacency": compact_adj,
        "poly_edges": compact_poly_edges,
        "poly_adjacency": compact_poly_adj,
        "angle_triplets": compact_triplets,
        "dihedrals": compact_dihedrals,
    }


def _fmt_dur(seconds):
    """Compact human-readable duration, e.g. '45s', '3m12s', '2h41m'."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _init_cgv4_worker():
    """Per-worker init: silence pymatgen's benign CIF-parse UserWarning (e.g.
    'N fractional coordinates rounded to ideal values to avoid ... finite
    precision'), which is harmless but would otherwise print once per structure
    and bury the progress output during a large build."""
    import warnings
    warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF",
                            category=UserWarning)


def _process_cgv4_row(task):
    """Worker: build + compact + write one graph JSON.

    task = (cif_id, cif_path, graph_path). Returns a status tuple
    (graph_path, kind, cif_id, msg) where kind is "ok" | "build_fail" |
    "post_fail". Defined at module level so it is picklable by multiprocessing.
    """
    cif_id, cif_path, graph_path = task
    from database.crystal_graph_v4_import import build_crystal_graph_from_cif
    try:
        # compute_spacegroup=False: the compact output drops the metadata block, so
        # symmetry analysis is wasted work here — and spglib floods stderr / can
        # wedge workers on distorted structures.
        graph = build_crystal_graph_from_cif(cif_path, compute_spacegroup=False)
    except Exception as exc:
        return (graph_path, "build_fail", cif_id, f"{type(exc).__name__}: {exc}")
    return _compact_and_write(graph, graph_path, cif_id)


def _compact_and_write(graph, graph_path, item_id):
    """Compact a full graph and write its JSON ATOMICALLY (tmp + os.replace).

    Shared post-build tail of both build workers. The atomic rename guarantees
    that a file at graph_path is always a complete JSON: a SIGKILL (e.g. SLURM
    walltime) mid-write leaves only a .tmp orphan, never a truncated .json — so
    the resumable builds' exists()-based skip can trust what it finds, and
    readers never hit JSONDecodeError on a half-written graph.
    """
    tmp_path = graph_path + ".tmp"
    try:
        compact = _compact_v4_graph(graph)
        with open(tmp_path, "w") as f:
            json.dump(compact, f)
        os.replace(tmp_path, graph_path)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return (graph_path, "post_fail", item_id, f"post-processing: {type(exc).__name__}: {exc}")
    return (graph_path, "ok", item_id, None)


def _process_cgv4_structure_row(task):
    """Worker: build + compact + write one graph JSON from an in-memory structure.

    Same contract as ``_process_cgv4_row`` but the input is a serialized pymatgen
    Structure dict (an MPtrj frame) instead of a CIF path, so no CIF round-trip.
    task = (frame_id, structure_dict, graph_path). Returns the same status tuple
    (graph_path, kind, frame_id, msg). Module-level so multiprocessing can pickle it.
    """
    frame_id, structure_dict, graph_path = task
    from database.crystal_graph_v4_import import build_crystal_graph_from_structure
    from pymatgen.core.structure import Structure
    try:
        structure = Structure.from_dict(structure_dict)
        # compute_spacegroup=False: metadata-only field the compact output drops;
        # spglib is also unreliable/slow on off-equilibrium MPtrj frames.
        graph = build_crystal_graph_from_structure(structure, compute_spacegroup=False)
    except Exception as exc:
        return (graph_path, "build_fail", frame_id, f"{type(exc).__name__}: {exc}")
    return _compact_and_write(graph, graph_path, frame_id)


def _write_index_files(index_rows, output_index):
    """Write the index pickle + csv for a list of row dicts.

    Shared by the CIF-sourced and structure-sourced builders so the index schema
    (column order, optional mp_id, target-column union) can never drift between
    them. Columns: legacy core first, then mp_id when any row carries it, then
    every recognized target column that appeared in any source.
    """
    present_targets = [c for c in KNOWN_TARGET_COLUMNS
                       if any(c in r for r in index_rows)]
    has_mp_id = any("mp_id" in r for r in index_rows)
    columns = (["id", "value", "graph_path", "label"]
               + (["mp_id"] if has_mp_id else []) + present_targets)
    index_df = pd.DataFrame(index_rows, columns=columns)
    index_df.to_pickle(output_index + ".pickle")
    index_df.to_csv(output_index + ".csv", index=False)
    return index_df


def generate_CGv4_DB(data_files: list, output_dir='database/datafiles/MP/graphs_v4',
                     output_index='database/datafiles/MP/SC_MP_V4', has_header=False,
                     limit=None, n_workers=None):
    """Pre-compute crystal_graph_v4 graphs for each material and store as compact JSON files.

    Creates one JSON per material in output_dir, plus an index pickle/csv at output_index.
    Resumable: skips any material whose JSON already exists.
    Failed structures are logged to output_dir/failed.txt and excluded from the index.

    Builds run in parallel across CIFs (each is independent); the main process
    writes the index. Output is identical to the serial version.

    Parameters
    ----------
    data_files : list of [csv_path, cif_dir] pairs
    output_dir : directory to write per-material JSON graph files
    output_index : path prefix for the index pickle/csv (appends .pickle / .csv)
    has_header : whether the source CSVs have a 'cif'/'tc' header row
    limit : if set, only process the first N rows per source (for testing)
    n_workers : number of worker processes (default: os.cpu_count()).
    """
    import multiprocessing as mp
    from collections import defaultdict

    os.makedirs(output_dir, exist_ok=True)
    failed_log = os.path.join(output_dir, "failed.txt")

    # Scan all source rows first, separating: already-built (index directly),
    # missing-CIF (failure), and to-build (dispatch to workers). Builds are
    # deduplicated by graph_path so the same material isn't rebuilt twice when
    # it appears in multiple sources.
    rows_for_path = defaultdict(list)   # graph_path -> [index_row, ...]
    build_tasks = {}                    # graph_path -> (cif_id, cif_path), first wins
    missing_cif = {}                    # graph_path -> cif_id
    existing_paths = set()

    for data_file in data_files:
        csv_path, cif_dir, label = _unpack_source(data_file)
        df = _load_id_prop(csv_path, has_header)
        if limit:
            df = df.head(limit)
        # Every recognized target column this source carries (e.g. just 'tc', or
        # 'e_above_hull' + 'formation_energy_per_atom'). All are written to the
        # index; the MPNN picks one at train time via `target_column`.
        target_cols = [c for c in KNOWN_TARGET_COLUMNS if c in df.columns]
        for _, row in df.iterrows():
            cif_id = row['cif']
            graph_path = os.path.join(output_dir, cif_id + ".json")
            rec = {"id": cif_id, "graph_path": graph_path, "label": label}
            for c in target_cols:
                rec[c] = row[c]
            # Legacy `value` column: mirrors 'tc' only (the historical single
            # target). It is deliberately left empty for tc-less sources (e.g. the
            # energy dataset) so a consumer that forgets to set `target_column`
            # fails loudly instead of silently regressing on an arbitrary target.
            rec["value"] = row["tc"] if "tc" in target_cols else None
            rows_for_path[graph_path].append(rec)

            if os.path.exists(graph_path):
                existing_paths.add(graph_path)
                continue
            if graph_path in build_tasks or graph_path in missing_cif:
                continue
            cif_path = os.path.join(cif_dir, cif_id)
            if os.path.exists(cif_path):
                build_tasks[graph_path] = (cif_id, cif_path)
            else:
                missing_cif[graph_path] = cif_id

    tasks = [(cif_id, cif_path, gp) for gp, (cif_id, cif_path) in build_tasks.items()]
    total = len(tasks)
    n_workers = (os.cpu_count() or 1) if n_workers is None else max(1, int(n_workers))
    n_workers = min(n_workers, total) if total else 1
    print(f"  {len(existing_paths)} already cached; {total} to build "
          f"on {n_workers} worker(s)")

    failed_lines = [f"{cif_id}\tCIF not found\n" for cif_id in missing_cif.values()]
    built_ok = set()

    if total:
        progress = {"done": 0}
        start = time.time()

        def _consume(results_iter):
            for graph_path, kind, cif_id, msg in results_iter:
                progress["done"] += 1
                d = progress["done"]
                if d % 50 == 0 or d == total:
                    elapsed = time.time() - start
                    rate = d / elapsed if elapsed > 0 else 0.0
                    eta = (total - d) / rate if rate > 0 else 0.0
                    print(f"  {d}/{total} ({100 * d / total:.1f}%) | "
                          f"elapsed {_fmt_dur(elapsed)} | ETA {_fmt_dur(eta)} | "
                          f"{rate:.1f} graphs/s")
                if kind == "ok":
                    built_ok.add(graph_path)
                else:
                    failed_lines.append(f"{cif_id}\t{msg}\n")

        if n_workers == 1:
            _init_cgv4_worker()   # suppress the CIF-parse warning in this process too
            _consume(map(_process_cgv4_row, tasks))
        else:
            # Context-managed pool so workers are always cleaned up, including
            # on KeyboardInterrupt / exception mid-build. The full iterator is
            # consumed inside the block, so all tasks finish before exit. Each
            # worker silences the benign pymatgen CIF-parse warning on startup.
            with mp.Pool(processes=n_workers, initializer=_init_cgv4_worker) as pool:
                _consume(pool.imap_unordered(_process_cgv4_row, tasks))

    # Emit one index row per source row whose graph file now exists.
    index_rows = []
    for gp, rows in rows_for_path.items():
        if gp in existing_paths or gp in built_ok:
            index_rows.extend(rows)

    if failed_lines:
        with open(failed_log, "a") as f:
            f.writelines(failed_lines)

    _write_index_files(index_rows, output_index)
    print(f"Done. {len(index_rows)} structures indexed, see {failed_log} for any failures.")


def generate_CGv4_DB_from_structures(record_iter, output_dir, output_index,
                                     label=1, limit=None, n_workers=None, chunksize=8):
    """Build compact cgv4 graphs from a STREAM of in-memory structures + labels.

    The CIF-based ``generate_CGv4_DB`` reads structures off disk; this variant
    consumes an iterator of records so a multi-GB source (e.g. the 12 GB MPtrj
    JSON) is never fully materialized. Each record is a dict:

        {"id": <unique frame id>, "structure": <pymatgen Structure as_dict()>,
         "formation_energy_per_atom": float, "energy_per_atom": float, ...}

    Every recognized target column present (see ``KNOWN_TARGET_COLUMNS``) is
    carried into the index pickle/csv so the MPNN picks one via `target_column`,
    exactly like the CIF path. Writes one graph JSON per frame into ``output_dir``
    and an index at ``output_index`` (+.pickle/.csv).

    Resumable: a frame whose graph JSON already exists is indexed but not rebuilt,
    so a re-run after a crash only does the remaining frames. Builds run across a
    process pool; structures are fed lazily (bounded memory) while the lightweight
    index rows accumulate in RAM. Duplicate ids are skipped (first wins).

    Parameters
    ----------
    record_iter : iterable of record dicts (typically a generator that streams)
    output_dir  : directory for per-frame JSON graphs
    output_index: path prefix for the index pickle/csv
    label       : value written to the index's `label` column. Default 1 to match
                  the energy datasets (MP_Energy): under `SC_to_non_SC_ratio=inf`
                  the split keeps only label==1 rows and drops label==0, so an
                  energy-regression dataset must be label 1 or every frame is
                  excluded from training.
    limit       : if set, stop after streaming this many (unique) frames — for tests
    n_workers   : pool size (default os.cpu_count())
    chunksize   : tasks dispatched per worker hand-off (throughput knob)
    """
    import multiprocessing as mp

    os.makedirs(output_dir, exist_ok=True)
    failed_log = os.path.join(output_dir, "failed.txt")

    index_rows = []      # lightweight rows (NO structure) -> the index at the end
    ok_paths = set()     # graph_paths that exist (pre-existing or freshly built)
    seen_ids = set()
    counters = {"streamed": 0, "skipped_existing": 0, "built": 0, "dup": 0,
                "failed": 0}
    start = time.time()

    # Failures are appended + flushed AS THEY HAPPEN (not buffered to the end):
    # a cancelled/killed job must still leave the failure reasons on disk — the
    # first cluster run was cancelled and lost all ~116k failure records because
    # they only existed in memory. Writes are guarded so a full disk can't crash
    # the run via its own error log.
    failed_fh = open(failed_log, "a")

    def _record_failure(fid, msg):
        counters["failed"] += 1
        try:
            failed_fh.write(f"{fid}\t{msg}\n")
            failed_fh.flush()
        except OSError:
            pass

    # Resume detection: ONE directory scan up front instead of a per-frame
    # os.path.exists — 1.6M individual stat() calls against a GPFS directory cost
    # tens of minutes of metadata traffic in the single-threaded feeder; a scandir
    # snapshot costs seconds. (Atomic .tmp orphans from a killed run are excluded
    # by the suffix check and get overwritten harmlessly.)
    existing_files = {e.name for e in os.scandir(output_dir)
                      if e.name.endswith(".json")}
    if existing_files:
        print(f"  resume: {len(existing_files)} graphs already on disk", flush=True)

    # Generator the pool's feeder thread drains: it both yields build tasks and
    # records the lightweight index row for every frame (built or already cached).
    def task_gen():
        for rec in record_iter:
            if limit and counters["streamed"] >= limit:
                break
            fid = str(rec["id"])
            if fid in seen_ids:                       # guard against any duplicate frame id
                counters["dup"] += 1
                continue
            seen_ids.add(fid)
            counters["streamed"] += 1
            graph_path = os.path.join(output_dir, fid + ".json")
            row = {"id": fid, "graph_path": graph_path, "label": label, "value": None}
            # Parent material id: REQUIRED for leakage-free train/val/test splits.
            # Frames of one trajectory are near-duplicates, so the split must group
            # by material (CIFDataV4 split_by="material"); the frame-id prefix is
            # NOT a reliable parent (MPtrj has frames filed under a different
            # mp_id), hence an explicit column.
            if rec.get("mp_id") is not None:
                row["mp_id"] = str(rec["mp_id"])
            for c in KNOWN_TARGET_COLUMNS:
                if rec.get(c) is not None:
                    row[c] = float(rec[c])
            index_rows.append(row)
            if fid + ".json" in existing_files:       # resume: already built
                ok_paths.add(graph_path)
                counters["skipped_existing"] += 1
                continue
            yield (fid, rec["structure"], graph_path)

    n_workers = (os.cpu_count() or 1) if n_workers is None else max(1, int(n_workers))

    # Systematic-storage-failure breaker: legit per-structure failures are fine,
    # but once writes start failing with quota/disk errors EVERY graph fails —
    # the first cluster run burned 2h+ churning ~116k such failures. Abort fast
    # with a clear message instead (the index for what DID build is still written).
    storage_strikes = {"n": 0}
    STORAGE_ERRORS = ("Disk quota exceeded", "No space left on device")

    def _consume(results_iter):
        for graph_path, kind, fid, msg in results_iter:
            if kind == "ok":
                ok_paths.add(graph_path)
                counters["built"] += 1
                storage_strikes["n"] = 0
            else:
                _record_failure(fid, msg)
                if any(e in (msg or "") for e in STORAGE_ERRORS):
                    storage_strikes["n"] += 1
                    if storage_strikes["n"] >= 25:
                        raise RuntimeError(
                            "Aborting: 25 consecutive storage failures "
                            f"(last: {msg}). The output filesystem is full or "
                            "over quota — free space / raise the quota, then "
                            "re-run (resumable; built graphs are kept).")
            done = counters["built"] + counters["failed"]
            if done % 200 == 0:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0.0
                print(f"  built {counters['built']} | failed {counters['failed']} | "
                      f"streamed {counters['streamed']} | {rate:.1f} graphs/s | "
                      f"elapsed {_fmt_dur(elapsed)}", flush=True)

    print(f"  building MPtrj graphs on {n_workers} worker(s) (streaming; resumable)...")
    try:
        if n_workers == 1:
            _init_cgv4_worker()
            _consume(map(_process_cgv4_structure_row, task_gen()))
        else:
            with mp.Pool(processes=n_workers, initializer=_init_cgv4_worker) as pool:
                _consume(pool.imap_unordered(_process_cgv4_structure_row, task_gen(),
                                             chunksize=chunksize))
    except BaseException:
        completed = False
        raise
    else:
        completed = True
    finally:
        failed_fh.close()
        # Write an index even when aborting mid-run (e.g. the storage breaker) so
        # partial progress is inspectable — but an INCOMPLETE run must never
        # clobber a complete existing index (a re-run that dies after streaming a
        # few thousand frames would otherwise silently shrink a 1.5M-row index).
        # Incomplete runs write to <output_index>.partial.* instead. Guarded: on a
        # full disk this write can itself fail, and that must not mask the
        # original error.
        out_prefix = output_index if completed else output_index + ".partial"
        status = ("complete" if completed
                  else "PARTIAL — kept separate from any existing full index")
        try:
            final_rows = [r for r in index_rows if r["graph_path"] in ok_paths]
            _write_index_files(final_rows, out_prefix)
            print(f"Index written ({status}): "
                  f"{out_prefix}.pickle — {len(final_rows)} frames "
                  f"({counters['skipped_existing']} pre-existing, "
                  f"{counters['built']} built this run, {counters['failed']} failed, "
                  f"{counters['dup']} duplicate ids skipped). "
                  f"See {failed_log} for any failures.", flush=True)
        except OSError as exc:
            print(f"WARNING: could not write index {out_prefix}.pickle: {exc}",
                  flush=True)
    return output_index + ".pickle"


def generate_Basic_DB(data_files: list, output_file='database/datafiles/MP/SC_MP_basic', parallel=False, timing=False,
                      batch_size=256, has_header=True, limit=None):
    """

    :param batch_size:
    :param timing:
    :param parallel:
    :param data_files: list of lists : [[data_file1.csv, cif_locs1], ...]
    :param output_file: output for the final DB
    :param has_header: whether the source CSVs have a 'cif'/'tc' header row
    :param limit: if set, only process the first ``limit`` rows of each CSV
    :return:
    """
    start = time.time()
    all_rows = []  # accumulate dict rows (id, value, struc_dict, label, +targets), build df once

    if not parallel:
        for data_file in data_files:
            csv_path, cif_dir, label = _unpack_source(data_file)
            df = _load_id_prop(csv_path, has_header)
            if limit:
                df = df.head(limit)

            Proc_Basic_Batch(df, all_rows, cif_dir, 0, label=label)
    else:
       for data_file in data_files:
            csv_path, cif_dir, label = _unpack_source(data_file)
            df = _load_id_prop(csv_path, has_header)
            if limit:
                df = df.head(limit)

            t1 = time.time()
            threads = []
            sub_lists = []
            errors = []  # threads can't propagate exceptions to the joiner; collect them

            def _worker(frame, out, cdir, tnum, lbl):
                try:
                    Proc_Basic_Batch(frame, out, cdir, tnum, label=lbl)
                except Exception as e:  # noqa: BLE001 - re-raised in the main thread below
                    errors.append(e)

            # Split into ~batch_size-row chunks with iloc, NOT np.array_split:
            # under numpy 2.x / pandas 3.0 np.array_split(df, ...) converts the
            # DataFrame to a bare ndarray (dropping .columns), which breaks the
            # per-chunk worker. iloc slicing keeps each chunk a real DataFrame.
            for start_i in range(0, len(df), batch_size):
                sub_frame = df.iloc[start_i:start_i + batch_size]
                sub_rows = []  # each thread appends to its own list (no shared state)
                sub_lists.append(sub_rows)
                t = threading.Thread(target=_worker,
                                     args=(sub_frame, sub_rows, cif_dir, len(sub_lists), label, ))
                t.start()
                threads.append(t)
            print("time to start threads: " + str(round(time.time() - t1, 1)) + " seconds")
            t1 = time.time()
            for thread in threads:
                thread.join()
            print("time waiting for kids " + str(round(time.time() - t1, 1)) + " seconds")
            # Surface any worker failure instead of silently writing a partial DB.
            if errors:
                raise errors[0]
            for sub_rows in sub_lists:
                all_rows.extend(sub_rows)

    # Stable column order: legacy core columns first, then every target column any
    # source carried (union). Rows are dicts; pandas fills missing keys with NaN.
    # Matches generate_CGv4_DB's index layout so CIFData can select via target_column.
    present_targets = [c for c in KNOWN_TARGET_COLUMNS if any(c in r for r in all_rows)]
    columns = ["id", "value", "struc_dict", "label"] + present_targets
    outdf = pd.DataFrame(all_rows, columns=columns)
    if len(outdf) == 0:
        raise ValueError(
            f"Basic DB build produced 0 rows for {output_file} — nothing was parsed. "
            "Check the source CSV path, --has-header, and the cif directory.")
    print(outdf.shape)
    outdf.to_pickle(output_file + ".pickle")
    outdf.to_csv(output_file + ".csv")
    if timing:
        print(
            "database constructions time: " + str(round(time.time() - start, 1)) + " with parallel = " + str(parallel))