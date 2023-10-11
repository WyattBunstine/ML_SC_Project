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
import csv
import torch
import matplotlib.pyplot as plt

df = pd.read_csv("CNN/test_result.csv",header=None)
plt.scatter(df[1].tolist(), df[2].tolist())
plt.xlabel("actual")
plt.ylabel("predicted")
plt.show()


