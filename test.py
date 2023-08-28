import json
from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pymatgen
from pymatgen import core
from pymatgen.symmetry.kpath import KPathSeek
import numpy as np
import pyxtal
import pymatgen
import pandas as pd

# specify the path of an experimental structure
df = pd.read_csv("database/all_data.csv")
API_KEY = "<MP_API_KEY>"
# inputs
with MPRester(API_KEY) as mpr:
    MPIDS = []
    for index, row in df.iterrows():
        if row['materials_project_id'] not in MPIDS:
            MPIDS.append(row['materials_project_id'])
        # save the relevent data to a dict
        if len(MPIDS) == 10:
            data = mpr.summary.search(material_ids=MPIDS)
            for mat in data:
                mat.structure.to("database/data/"+str(mat.material_id)+".cif")
            MPIDS = []
    data = mpr.summary.search(material_ids=MPIDS)
    for mat in data:
        mat.structure.to("database/data/" + str(mat.material_id) + ".cif")
    MPIDS = []


