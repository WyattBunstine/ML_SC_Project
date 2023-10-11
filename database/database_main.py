from pymatgen.ext.matproj import MPRester
import main
import csv
import pandas as pd
import pymatgen
import numpy as np
import json
API_KEY = "<MP_API_KEY>"

def rebuild_database():
    #check if the debug database should be used
    if not main.DEBUG:
        #add some code here to pull all the info from materials proj
        with MPRester(API_KEY) as mpr:
            1+1
    #build a debug database to use with model to check functionallity
    else:
        1+1

def gen_CGCNN_DB():
    df = pd.read_csv("database/MP/3DSC_MP.csv")
    cifs = df["cif"].tolist()
    cifs_new = []
    for cif in cifs:
        cifs_new.append(cif[19:])

    with open('database/MP/id_prop.pickle', 'w') as f:
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


df = pd.read_csv("database/MP/3DSC_MP.csv")
df = df.drop(df.columns.difference(["tc", "cif"]),axis=1)
newdf = pd.DataFrame(columns=["id", "value", "struc_dict"])


for index, row in df.iterrows():
    structure = pymatgen.core.structure.Structure.from_file("database/MP/cifs/" + row['cif'][19:])
    newdf.loc[len(newdf.index)] = (row['cif'][19:], row['tc'], structure.as_dict())


newdf.to_pickle('database/MP/id_prop.pickle')