import itertools

from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifWriter
import main
import csv
import pandas as pd
import pymatgen
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import *
import numpy as np
import json
import scipy
import collections

API_KEY = "<MP_API_KEY>"

def gen_dataset(min_band_gap = 1.0, prop_file="~/PycharmProjects/ML_SC_Project/database/Non_SC_DB_MP/Non_SC.csv",cif_loc = "~/PycharmProjects/ML_SC_Project/database/Non_SC_DB_MP/cifs/"):
    with MPRester(API_KEY) as mpr:
        data = mpr.materials.summary.search(band_gap=[min_band_gap, 1000], all_fields=False, fields=["composition", "material_id", "structure"])
        with open(prop_file, "w+") as file:
            for mat in data:
                file.write(mat.material_id+".cif,0.0\n")
                w = CifWriter(mat.structure)
                w.write_file(cif_loc + mat.material_id + '.cif')


