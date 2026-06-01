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


def _load_id_prop(csv_path, has_header=True):
    """Read an id->property CSV into a DataFrame with 'cif' and 'tc' columns.

    has_header=True  : the CSV has a header row containing (at least) 'cif' and
                       'tc' columns, e.g. 3DSC_MP.csv. All other columns are dropped.
    has_header=False : the CSV has no header and two columns ordered
                       (cif_filename, tc), e.g. id_prop.csv.
    """
    if has_header:
        df = pd.read_csv(csv_path)
        return df.drop(df.columns.difference(["tc", "cif"]), axis=1)
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


def generate_atom_init(output_file='database/atom_init.json', max_z=85):
    """Build the per-element feature file consumed by the CNN's AtomInitializer.

    Writes a JSON object mapping each atomic number Z (1 .. max_z - 1) to a feature
    vector: [Z, block, valence, atomic_radius, electron_affinity, ionization_energy,
    electronegativity, electron_affinity].
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
    """Parse each CIF in ``df`` and append an (id, value, struc_dict, label) tuple
    to the list ``out_rows``. Appending to a list is O(1); the DataFrame is built
    once by the caller. (The previous ``DataFrame.loc[...] = row`` enlargement was
    O(n^2) and became pathologically slow on the ~61k-row combined dataset.)"""
    warnings.showwarning = customwarn
    t1 = time.time()
    total = len(df)
    for j, (index, row) in enumerate(df.iterrows(), 1):
        structure = pymatgen.core.structure.Structure.from_file(cif_loc + row['cif'])
        out_rows.append((row['cif'], row['tc'], structure.as_dict(), label))
        if j % 500 == 0:
            print("  thread " + str(thread_num) + ": " + str(j) + "/" + str(total) +
                  " (" + str(round(100 * j / total, 1)) + "%)  " +
                  str(round(time.time() - t1, 1)) + "s")
    print("Thread " + str(thread_num) + " time: " + str(round(time.time()-t1, 4)) + " seconds.")


def generate_CGv4_DB(data_files: list, output_dir='database/MP/graphs_v4',
                     output_index='database/MP/id_prop_v4', has_header=False, limit=None):
    """Pre-compute crystal_graph_v4 graphs for each material and store as compact JSON files.

    Creates one JSON per material in output_dir, plus an index pickle/csv at output_index.
    Resumable: skips any material whose JSON already exists.
    Failed structures are logged to output_dir/failed.txt and excluded from the index.

    Parameters
    ----------
    data_files : list of [csv_path, cif_dir] pairs
    output_dir : directory to write per-material JSON graph files
    output_index : path prefix for the index pickle/csv (appends .pickle / .csv)
    has_header : whether the source CSVs have a 'cif'/'tc' header row
    limit : if set, only process the first N rows per source (for testing)
    """
    from database.crystal_graph_v4_import import build_crystal_graph_from_cif
    from pymatgen.core.periodic_table import Element as PmgElement

    os.makedirs(output_dir, exist_ok=True)
    failed_log = os.path.join(output_dir, "failed.txt")

    index_rows = []

    for data_file in data_files:
        csv_path, cif_dir, label = _unpack_source(data_file)
        df = _load_id_prop(csv_path, has_header)
        if limit:
            df = df.head(limit)

        total = len(df)
        for j, (_, row) in enumerate(df.iterrows()):
            cif_id = row['cif']
            tc = row['tc']

            if j % 50 == 0:
                print(f"  {j}/{total} ({100*j/total:.1f}%)")

            graph_path = os.path.join(output_dir, cif_id + ".json")

            # Resumable: skip if already processed
            if os.path.exists(graph_path):
                index_rows.append({"id": cif_id, "value": tc, "graph_path": graph_path, "label": label})
                continue

            cif_path = os.path.join(cif_dir, cif_id)
            if not os.path.exists(cif_path):
                with open(failed_log, "a") as f:
                    f.write(f"{cif_id}\tCIF not found\n")
                continue

            try:
                graph = build_crystal_graph_from_cif(cif_path)
            except Exception as exc:
                with open(failed_log, "a") as f:
                    f.write(f"{cif_id}\t{type(exc).__name__}: {exc}\n")
                continue

            # Build compact representation — only fields used as features
            try:
                compact_nodes = []
                for node in graph["nodes"]:
                    z = PmgElement(node["element"]).Z
                    ion_role_map = {"cation": 1, "anion": -1, "neutral": 0}
                    hist = node.get("sharing_mode_hist_core") or {"corner": 0, "edge": 0, "face": 0, "other": 0}
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
                    })

                compact_edges = []
                for edge in graph["edges"]:
                    compact_edges.append({
                        "id": edge["id"],
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

                # Adjacency: node_id -> [(edge_id, neighbor_id), ...]
                compact_adj = {
                    str(nid): [(eid, nbr) for (eid, nbr, _) in neighbors]
                    for nid, neighbors in graph["adjacency"].items()
                }

                compact = {"nodes": compact_nodes, "edges": compact_edges, "adjacency": compact_adj}

                with open(graph_path, "w") as f:
                    json.dump(compact, f)

                index_rows.append({"id": cif_id, "value": tc, "graph_path": graph_path, "label": label})

            except Exception as exc:
                with open(failed_log, "a") as f:
                    f.write(f"{cif_id}\tpost-processing: {type(exc).__name__}: {exc}\n")
                if os.path.exists(graph_path):
                    os.remove(graph_path)

    index_df = pd.DataFrame(index_rows, columns=["id", "value", "graph_path", "label"])
    index_df.to_pickle(output_index + ".pickle")
    index_df.to_csv(output_index + ".csv", index=False)
    print(f"Done. {len(index_rows)} structures indexed, see {failed_log} for any failures.")


def generate_Basic_DB(data_files: list, output_file='database/id_prop_basic', parallel=False, timing=False,
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
    all_rows = []  # accumulate (id, value, struc_dict, label) tuples, build df once

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
            for sub_frame in np.array_split(df, max(1, int(len(df) / batch_size))):
                sub_rows = []  # each thread appends to its own list (no shared state)
                sub_lists.append(sub_rows)
                t = threading.Thread(target=Proc_Basic_Batch,
                                     args=(sub_frame, sub_rows, cif_dir, len(sub_lists), label, ))
                t.start()
                threads.append(t)
            print("time to start threads: " + str(round(time.time() - t1, 1)) + " seconds")
            t1 = time.time()
            for thread in threads:
                thread.join()
            print("time waiting for kids " + str(round(time.time() - t1, 1)) + " seconds")
            for sub_rows in sub_lists:
                all_rows.extend(sub_rows)

    outdf = pd.DataFrame(all_rows, columns=["id", "value", "struc_dict", "label"])
    print(outdf.shape)
    outdf.to_pickle(output_file + ".pickle")
    outdf.to_csv(output_file + ".csv")
    if timing:
        print(
            "database constructions time: " + str(round(time.time() - start, 1)) + " with parallel = " + str(parallel))