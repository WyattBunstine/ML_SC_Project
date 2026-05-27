import time
import warnings
import threading

from pymatgen.ext.matproj import MPRester
import pandas as pd
import pymatgen
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import *
import numpy as np
import json
import scipy

API_KEY = "zAtoaKbzoIxH07M5EFsQZbrO9a0GRvz1"

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


def generate_CE_DB(data_files: list, output_file='database/id_prop_ce', has_header=True, limit=None):
    """

    :param data_files: list of lists : [[data_file1.csv, cif_locs1], ...]
    :param output_file: output for the final DB
    :param has_header: whether the source CSVs have a 'cif'/'tc' header row
    :param limit: if set, only process the first ``limit`` rows of each CSV
    :return:
    """
    warnings.filterwarnings("ignore")
    outdf = pd.DataFrame(columns=["id", "value", "struc_dict", "ce"])
    knownCes = {}
    num_ce = 0

    k = 0
    for data_file in data_files:
        k += 1
        df = _load_id_prop(data_file[0], has_header)
        if limit:
            df = df.head(limit)

        end = df.shape[0]
        j = 0
        for index, row in df.iterrows():
            j += 1
            if j % 50 == 0:
                print(str(j * 100 / end)[0:5] + "% finished of dataset " + str(k) + " of " + str(
                    len(data_files)) + " datasets")
            if j % 250 == 0:
                outdf.to_pickle(output_file + ".pickle")
            if row['cif'] not in outdf["id"].tolist():
                structure = pymatgen.core.structure.Structure.from_file(data_file[1] + row['cif'])

                bva = BVAnalyzer()
                try:
                    vals = bva.get_valences(structure=structure)
                except ValueError:
                    vals = np.zeros(len(structure.sites))
                valence = []
                for val in vals:
                    if type(val) == list:
                        valence.append(np.average(val))
                    else:
                        valence.append(val)
                try:
                    coord_env = pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder.LocalGeometryFinder().compute_coordination_environments(
                        structure, only_cations=False, valences=valence)
                except (scipy.spatial._qhull.QhullError, RuntimeError):
                    coord_env = [[] for _ in range(len(structure.sites))]
                # print(periodic_sites['sites'][0])
                coordination_environment = []
                for i in range(len(coord_env)):
                    if len(coord_env[i]) > 0:
                        ce = coord_env[i][0]['ce_symbol']
                    else:
                        ce = 0
                    if not ce in knownCes.keys():
                        num_ce += 1
                        knownCes[ce] = num_ce
                    coordination_environment.append(knownCes[ce])

                outdf.loc[len(outdf.index)] = (row['cif'], row['tc'], structure.as_dict(), coordination_environment)

    outdf.to_pickle(output_file + ".pickle")
    outdf.to_csv(output_file + ".csv")


def Proc_Basic_Batch(df, outdf, cif_loc, thread_num):
    warnings.showwarning = customwarn
    t1 = time.time()
    for index, row in df.iterrows():
        structure = pymatgen.core.structure.Structure.from_file(cif_loc + row['cif'])
        outdf.loc[len(outdf.index)] = (row['cif'], row['tc'], structure.as_dict())
    print("Thread " + str(thread_num) + " time: " + str(round(time.time()-t1, 4)) + " seconds.")


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
    outdf = pd.DataFrame(columns=["id", "value", "struc_dict"])

    if not parallel:
        for data_file in data_files:
            df = _load_id_prop(data_file[0], has_header)
            if limit:
                df = df.head(limit)

            Proc_Basic_Batch(df, outdf, data_file[1], 0)
            '''
            for index, row in df.iterrows():
                structure = pymatgen.core.structure.Structure.from_file(data_file[1] + row['cif'])
                outdf.loc[len(outdf.index)] = (row['cif'], row['tc'], structure.as_dict())'''
    else:
       for data_file in data_files:
            df = _load_id_prop(data_file[0], has_header)
            if limit:
                df = df.head(limit)

            t1 = time.time()
            threads = []
            sub_frames = []
            for sub_frame in np.array_split(df, int(len(df)/batch_size)):
                sub_out_frame = pd.DataFrame(columns=["id", "value", "struc_dict"])
                sub_frames.append(sub_out_frame)
                t = threading.Thread(target=Proc_Basic_Batch,
                                     args=(sub_frame, sub_out_frame, data_file[1], len(sub_frames), ))
                t.start()
                threads.append(t)
            print("time to start threads: " + str(round(time.time() - t1, 1)) + " seconds")
            t1 = time.time()
            for thread in threads:
                thread.join()
            print("time waiting for kids " + str(round(time.time() - t1, 1)) + " seconds")
            t1 = time.time()
            outdf = pd.concat(sub_frames, ignore_index=True)
            print("time writing output frame " + str(round(time.time() - t1, 1)) + " seconds")
    print(outdf.shape)
    outdf.columns = ["id", "value", "struc_dict"]
    outdf.to_pickle(output_file + ".pickle")
    outdf.to_csv(output_file + ".csv")
    if timing:
        print(
            "database constructions time: " + str(round(time.time() - start, 1)) + " with parallel = " + str(parallel))