import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("CNN/data/results_ce.csv", header=None)
data2 = pd.read_csv("CNN/data/results_no_ce.csv", header=None)

for  index, row in data.iterrows():
    if row[2] > 15 and row[1] < 5:
        print(row[0]+" pred: "+str(row[2]) + "   Target: "+str(row[1]))

MSE1 = np.sum(np.abs(data[1]-data[2])**2)/len(data[1])
MSE2 = np.sum(np.abs(data2[1]-data2[2])**2)/len(data2[1])

plt.scatter(data[1], data[2], label="CE MAE:"+str(MSE1))
plt.scatter(data2[1], data2[2], label="No_CE MAE:"+str(MSE2))
plt.xlim([5,100])
plt.xlabel("target")
plt.ylabel("prediction")
plt.legend()
plt.show()
