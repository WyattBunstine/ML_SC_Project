import sys
import CNN.CNN
import database.DBMain
import pickle
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #define the different things to be done
    if len(sys.argv)>0:
        #rebuild the database
        if "--rebuild_database" in sys.argv:
            database.DBMain.build_debug_database("debug_database.db")
        if "--reload_weights" in sys.argv:
            if sys.argv.index("--reload") + 1 >= len(sys.argv):
                print("file name not specified")
            else:
                file = sys.argv[sys.argv.index("--reload")+1]
                if not file[-4:] == ".wgt":
                    print("Weights file defined incorrectly")
                else:
                    print(file)
        if "--CNN" in sys.argv:
            if sys.argv.index("--CNN") > len(sys.argv)+2:
                print("wrong input length for --CNN")
            else:
                data_set = np.loadtxt(sys.argv[sys.argv.index("--CNN")+2], delimiter=",", dtype="float32")
                CNN.CNN.cnn_main(sys.argv[sys.argv.index("--CNN")+1], data_set)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
