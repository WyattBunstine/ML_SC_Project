import pickle
import numpy as np


def debug_objective_function(i):
    return i[0] * i[1] + i[5] * i[7] - i[7] * i[3] / i[6] + i[6] * i[8] * i[2] + i[0] * np.cos(i[9])


def build_debug_database(file, data_size=100000, input_size=10, output_type="class"):
    # if debug is true, make some dataset that is from some well-defined function
    data = []
    for i in np.arange(data_size):
        inputs = np.random.rand(input_size)
        data.append(np.append(inputs, debug_objective_function(inputs)))
    np.savetxt(file, np.array(data), delimiter=",")


def load():
    1
