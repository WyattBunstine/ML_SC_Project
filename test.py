import json
#from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pymatgen
from pymatgen import core
from pymatgen.symmetry.kpath import KPathSeek
import numpy as np
import pymatgen
import pymatgen.analysis.diffraction.xrd

struct = []
struct.append(pymatgen.core.structure.Structure.from_file("YbMgGaO4_VP.cif"))
struct.append(pymatgen.core.structure.Structure.from_file("YbMgGaO4_VPMg.cif"))
struct.append(pymatgen.core.structure.Structure.from_file("YbMgGaO4_VPGa.cif"))

xrd = pymatgen.analysis.diffraction.xrd.XRDCalculator()
fig = xrd.plot_structures(struct,annotate_peaks = None,ax_annotate=False)
fig.show()