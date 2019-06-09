import pandas as pd
import numpy as np

df = pd.read_csv("all_queues.tr")

numGRouter = np.sum(df['GfromRouter'] + df['RouterFromG'])

print(numGRouter / 5.0)


numGServer = np.sum(df['GfromServer'] + df['ServerFromG'])

print(numGServer / 5.0)


test = np.sum(df['ServerFromG']) / len(df['ServerFromG'])

print(test)
