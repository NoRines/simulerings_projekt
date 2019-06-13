import csv
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("1.tr")
data_arrs = pd.DataFrame()
data_arrs["1"] = data["GfromServer"]

for i in range(2,16):
    data = pd.read_csv(str(i)+".tr")
    data_arrs[i] = data["GfromServer"]
    
data_mean =  data_arrs.mean(axis=1)
data_std = data_arrs.std(axis=1)

data_arrs["mean"] = data_mean
data_arrs["stdev"] = data_std
data_arrs["time"] = data["time"]
print(data_arrs)

plt.errorbar(data_arrs["time"],data_arrs["mean"],data_arrs["stdev"],ecolor='black',color='white')
plt.xlabel('Simulation time (s)')
plt.ylabel('Packets in queue (pcs)')

plt.show()
