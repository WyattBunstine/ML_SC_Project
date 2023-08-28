import json
from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pymatgen
from pymatgen import core
from pymatgen.symmetry.kpath import KPathSeek
import numpy as np
import pyxtal
import pymatgen

# specify the path of an experimental structure
struc_file = "OrbitalOverlap/Cu4BiSe4I.cif"

xtal1 = pyxtal.pyxtal()
xtal1.from_seed(seed=struc_file, style='pyxtal')
#print(xtal1)

# visualize the structure
xtal1.show(supercell=(2,2,1))

