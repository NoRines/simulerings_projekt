import pandas as pd
import numpy as np

df = pd.read_csv("all_queues.tr")

numGRouter = np.sum(df['GfromRouter'] + df['RouterFromG'])

print(numGRouter / 5.0)


numGServer = np.sum(df['GfromServer'] + df['ServerFromG']) / 5.0

print(numGServer)


test = np.sum(df['ServerFromG']) / len(df['ServerFromG'])
#test = np.sum(df['ServerFromG']) / 5

print(test)
print(len(df['ServerFromG']))
