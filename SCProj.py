import sys
import CNN.CNN
import Transformer.Transformer
import database.DBMain
import pickle
import numpy as np
import json
from warnings import *

# Press the green button in the gutter to run the script.
def main():
    #define the different things to be done
    if len(sys.argv)>0:
        if not sys.argv[0][:,-4] == ".cfg":
            warn("valid config file not specified")
        else:
            config = json.load(sys.argv[0])
            if config["model"] == "CGCNN":
                CNN.main(config)
            if config["model"] == "transformer":
                Transformer.main(config)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
if __name__ == '__main__':
    main()