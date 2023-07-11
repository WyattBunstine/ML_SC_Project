import numpy as np
from pymatgen.symmetry import groups

symops = []
duplicates = []
for sgn in np.arange(1,231):
    sg = groups.SpaceGroup(groups.sg_symbol_from_int_number(sgn))
    for sym in sg.symmetry_ops:
        if sym not in symops:
            symops.append(sym)
        elif sym not in duplicates:
            duplicates.append(sym)
#there are 4425 symmetries among all space groups
#there are 941 unique symmetry operations
#368 of those only appear in one spacegroup
print(len(symops))
print(len(duplicates))

