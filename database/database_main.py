import itertools
import time
import warnings
import threading

from pymatgen.ext.matproj import MPRester
import main
import csv
import pandas as pd
import pymatgen
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import *
import numpy as np
import json
import scipy
import collections
from pymatgen.core.composition import Composition
from pymatgen.core.composition import Element
import os
from monty.fractions import gcd, gcd_float
from monty.json import MSONable
from monty.serialization import loadfn
from itertools import combinations_with_replacement, product
import multiprocessing

API_KEY = "<MP_API_KEY>"

def customwarn(message, category, filename, lineno, file=None, line=None):
    1+1 # sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))



def gen_CGCNN_DB():
    df = pd.read_csv("database/MP/3DSC_MP.csv")
    cifs = df["cif"].tolist()
    cifs_new = []
    for cif in cifs:
        cifs_new.append(cif[19:])

    with open('database/MP/id_prop_basic.pickle', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(cifs_new, df["tc"].tolist()))

    elements = {}
    for i in np.arange(1, 85):
        el = pymatgen.core.periodic_table.Element.from_Z(i)
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

    with open('atom_init.json', 'w') as f:
        json.dump(elements, f)


def generate_CE_DB(data_files: list, output_file='database/id_prop_ce'):
    """

    :param data_files: list of lists : [[data_file1.csv, cif_locs1], ...]
    :param output_file: output for the final DB
    :return:
    """
    warnings.filterwarnings("ignore")
    outdf = pd.DataFrame(columns=["id", "value", "struc_dict", "ce"])
    knownCes = {}
    num_ce = 0

    k = 0
    for data_file in data_files:
        k += 1
        df = pd.read_csv(data_file[0])
        df = df.drop(df.columns.difference(["tc", "cif"]), axis=1)

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
                      batch_size=256):
    """

    :param batch_size:
    :param timing:
    :param parallel:
    :param data_files: list of lists : [[data_file1.csv, cif_locs1], ...]
    :param output_file: output for the final DB
    :return:
    """
    start = time.time()
    outdf = pd.DataFrame(columns=["id", "value", "struc_dict"])

    if not parallel:
        for data_file in data_files:
            df = pd.read_csv(data_file[0])
            df = df.drop(df.columns.difference(["tc", "cif"]), axis=1)

            Proc_Basic_Batch(df, outdf, data_file[1], 0)
            '''
            for index, row in df.iterrows():
                structure = pymatgen.core.structure.Structure.from_file(data_file[1] + row['cif'])
                outdf.loc[len(outdf.index)] = (row['cif'], row['tc'], structure.as_dict())'''
    else:
       for data_file in data_files:
            df = pd.read_csv(data_file[0])
            df = df.drop(df.columns.difference(["tc", "cif"]), axis=1)

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


def generate_CE_DB_OLD():
    df = pd.read_csv("database/MP/3DSC_MP.csv")
    df = df.drop(df.columns.difference(["tc", "cif"]), axis=1)
    newdf = pd.DataFrame(columns=["id", "value", "struc_dict", "ce"])

    newdf = pd.read_pickle('database/MP/id_prop_basic.pickle')
    knownCes = {}
    num_ce = 0

    end = df.shape[0]
    j = 0
    for index, row in df.iterrows():
        j += 1
        if j % 50 == 0:
            print(str(j * 100 / end)[0:5] + "% finished")
        if j % 250 == 0:
            newdf.to_pickle('database/MP/id_prop_basic.pickle')
        if row['cif'][19:] not in newdf["id"].tolist():
            structure = pymatgen.core.structure.Structure.from_file("database/MP/cifs/" + row['cif'][19:])

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

            newdf.loc[len(newdf.index)] = (row['cif'][19:], row['tc'], structure.as_dict(), coordination_environment)

    newdf.to_pickle('database/MP/id_prop_basic.pickle')


def get_oxi_state_guesses(structure: pymatgen.core.structure.Structure, target_charge=0, charge_err=0.25,
                          all_oxi=False):
    """
    This is a modified version of the PyMatGen function to find oxidation states. This was written to take partial
    occupation into considerion. Simply weighted average of oxidation state between partial occuations
    :param all_oxi:
    :param charge_err:
    :param target_charge: the target charge of the structure. 0 is charge balanced
    :param structure: structure for which to guess states
    :return: tuples of oxidation state guesses
    """

    # Load prior probabilities of oxidation states, used to rank solutions
    ICSD_states = {}
    module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    all_data = loadfn(f"{module_dir}/../venv/lib/python3.11/site-packages/pymatgen/analysis/icsd_bv.yaml")
    total_data = {}
    for sp, data in all_data["occurrence"].items():
        if Species.from_str(sp).element in total_data.keys():
            total_data[Species.from_str(sp).element] += data
        else:
            total_data[Species.from_str(sp).element] = data
        if Species.from_str(sp).element in ICSD_states.keys():
            ICSD_states[Species.from_str(sp).element].append(Species.from_str(sp).oxi_state)
        else:
            ICSD_states[Species.from_str(sp).element] = [Species.from_str(sp).oxi_state]
    Composition.oxi_prob = {Species.from_str(sp): data / total_data[Species.from_str(sp).element] for sp, data in
                            all_data["occurrence"].items()}
    # for each element, determine all possible sum of oxidations
    site_combos = []
    structure.remove_oxidation_states()
    for site in structure.sites:
        species = site.species.as_dict()
        for specie in species.keys():
            el = Element(specie)
            oxids = list(el.oxidation_states)
            if all_oxi:
                oxids = list(el.oxidation_states) + list(set(ICSD_states[el]) - set(el.oxidation_states))
            probs = []
            for oxid in oxids:
                probs.append([str(el), oxid, species[specie]])
            site_combos.append(probs)
    print(len(site_combos))
    print(site_combos)
    combo_scores = {}
    for combo in itertools.product(*site_combos):
        if abs(sum([site[1] * site[2] for site in combo])) - target_charge < charge_err:
            combo_score = np.prod([Composition.oxi_prob.get(Species(site[0], site[1]), 0) * site[2] for site in combo])
            combo_scores[combo_score] = combo
    if len(combo_scores) == 0:
        if all_oxi == False:
            print("No common oxidation state combos found, trying all possible")
            return get_oxi_state_guesses(structure, target_charge=target_charge, charge_err=charge_err, all_oxi=True)
        else:
            return None
        # TODO make some catch for no possible oxidation state combos

    keys = list(combo_scores.keys())
    keys.sort()
    for combo in keys:
        prt = "{:.5f}".format(combo)
        print(prt + ":  " + str(combo_scores[combo]))
    # print(np.max(list(combo_scores.keys())))
    # print(combo_scores[np.max(list(combo_scores.keys()))])
    print(sum([site[1] * site[2] for site in combo_scores[np.max(list(combo_scores.keys()))]]))
    return combo_scores[np.max(list(combo_scores.keys()))]



'''        for data_file in data_files:
            df = pd.read_csv(data_file[0])
            df = df.drop(df.columns.difference(["tc", "cif"]), axis=1)
            for index, row in df.iterrows():
                outdf.loc[len(outdf.index)] = (row['cif'], row['tc'], data_file[1] + row['cif'])
        print("basic outdf time: " + str(round(time.time() - start, 1)) + " DB len: " + str(
            len(outdf)) + " with parallel = " + str(parallel))
        t1 = time.time()
        threads = []
        for i in range(0, len(outdf) - len(outdf) % batch_size, batch_size):
            t = threading.Thread(target=Proc_Basic_Batch, args=(outdf, i, batch_size))
            t.start()
            threads.append(t)
        t = threading.Thread(target=Proc_Basic_Batch,
                             args=(outdf, len(outdf) - len(outdf) % batch_size, len(outdf) % batch_size))
        t.start()
        threads.append(t)
        print("time to start threads: " + str(round(time.time() - t1, 1)) + " with parallel = " + str(parallel))
        t1 = time.time()
        for thread in threads:
            thread.join()
        print("time waiting for kids " + str(round(time.time() - t1, 1)) + " with parallel = " + str(parallel))'''