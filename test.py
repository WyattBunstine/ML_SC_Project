import json
import os
import time
import ctypes
import threading
from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pymatgen
from pymatgen import core, analysis
from pymatgen.symmetry.kpath import KPathSeek
import numpy as np
import pyxtal
import pymatgen
import pandas as pd
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import *
import csv
import torch
import matplotlib.pyplot as plt
import scipy
from database.database_main import *
import database.Non_SC_DB_MP.Download_MP_data as Download_MP_data


def get_valence(el, oxi):
    x = el.group
    val = x
    if 56 < el.Z < 71:
        val = el.Z - 56 - oxi
    elif 3 <= x <= 12:
        val = x - 3 - oxi
    elif x > 12:
        val = x - 13 - oxi
    return val


# structure = pymatgen.core.structure.Structure.from_file("database/MP/cifs/Ag0.1Ge2Pd1.9Sr1-MP-mp-978986-synth_doped.cif")
# structure = pymatgen.core.structure.Structure.from_file("ICSD_CollCode160675.cif")
def test_func(index, df):
    df.iloc[index] = [1, 2]

'''
df = pd.DataFrame([[2, 3], [3, 4], [3, 5], [3, 6]])
print("Original")
print(df)
test_func(0, df)
print("after 1")
print(df)
thread = threading.Thread(target=test_func, args=(1, df,))
thread.start()
thread.join()
print("final")
print(df)'''

#print(59753%2048)
# , ["database/Non_SC_DB_MP/Non_SC.csv","database/Non_SC_DB_MP/cifs/"]
generate_Basic_DB([["database/MP/id_prop.csv", "database/MP/cifs/"]],
                  output_file='database/id_prop_basic_test_parallel_small', timing=True, parallel=True, batch_size=1000)

1 / 0

df = pd.read_csv("database/MP/3DSC_MP.csv")
print(df["cif_before_synthetic_doping"])
print(df.columns.values)
1 / 0

structure = pymatgen.core.structure.Structure.from_file("database/EntryWithCollCode44602.cif")
print(get_oxi_state_guesses(structure))

1 / 0

df = pd.read_pickle('database/MP/id_prop_ce.pickle')
knownCes = {}
num_ce = 0

newdf = pd.DataFrame(columns=["id", "value", "struc_dict", "ce", "valence"])
end = df.shape[0]
j = 0
for index, row in df.iterrows():
    j += 1
    if j % 50 == 0:
        print(newdf.head)
        1 / 0
        print(str(j * 100 / end)[0:5] + "% finished")
    if j % 250 == 0:
        newdf.to_pickle('database/MP/id_prop_ce_val.pickle')
    # if row['id'] not in newdf["id"].tolist():
    structure = pymatgen.core.structure.Structure.from_file("database/MP/cifs/" + row['id'])
    valence = []
    structure.add_oxidation_state_by_guess()
    for site in structure.sites:
        oxistate = np.average([get_valence(specie.element, specie.oxi_state) for specie in site.species])

    newdf.loc[len(newdf.index)] = (row['id'], row['value'], structure.as_dict(), row["ce"], valence)

newdf.to_pickle('database/MP/id_prop_ce_val.pickle')
