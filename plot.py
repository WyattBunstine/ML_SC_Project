import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_RESULTS = "CNN/test_result.csv"


def plot_results(results_file=DEFAULT_RESULTS):
    """Scatter-plot CNN test predictions vs. targets.

    ``results_file`` is the headerless (cif_id, target, prediction) CSV written by
    CGCNNMain.py's test pass.
    """
    data = pd.read_csv(results_file, header=None)

    mse = np.sum(np.abs(data[1] - data[2])) / len(data[1])

    plt.scatter(data[1], data[2], label="MSE: " + str(mse))
    plt.plot([np.min(data[1]), np.max(data[1])], [np.min(data[1]), np.max(data[1])], color='black', linestyle='--')
    #plt.xlim([0, 100])
    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_results()
