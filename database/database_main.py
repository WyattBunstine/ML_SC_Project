from pymatgen.ext.matproj import MPRester
import main
API_KEY = "zAtoaKbzoIxH07M5EFsQZbrO9a0GRvz1"

def rebuild_database():
    #check if the debug database should be used
    if not main.DEBUG:
        #add some code here to pull all the info from materials proj
        with MPRester(API_KEY) as mpr:
            1+1
    #build a debug database to use with model to check functionallity
    else:
        1+1