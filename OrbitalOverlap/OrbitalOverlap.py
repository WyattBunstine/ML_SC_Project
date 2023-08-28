import pyxtal
import pymatgen

# specify the path of an experimental structure
cif_nacl = resource_filename("pyxtal", "database/cifs/NaCl.cif")
cif_aspirin = resource_filename("pyxtal", "database/cifs/aspirin.cif")

# load the structure from pyxtal

# if you load the atomic crystal
#xtal1 = pyxtal()
#xtal1.from_seed(seed = cif_nacl)

# to load a molecular crystal, also needs to specify the molecule tag
xtal1 = pyxtal(molecular=True)
xtal1.from_seed(seed=cif_aspirin, molecules=['aspirin'])
print(xtal1)

# visualize the structure
xtal1.show()