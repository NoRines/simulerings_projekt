import pandas as pd
import numpy as np

dff = pd.read_csv("ffile.csv")
dfg = pd.read_csv("gfile.csv")


ftimes = dff['Time'].tolist()
gtimes = dfg['Time'].tolist()

fsizes = dff['Length'].tolist()
gsizes = dfg['Length'].tolist()

print(len(ftimes))
print(len(gtimes))

totdelay = 0.0

for i in range(len(gtimes)):
    delay = float(gtimes[i]) - float(ftimes[i])
    if fsizes[i] != gsizes[i]:
        print('hej')
    print(delay)
    totdelay += delay


print(totdelay / len(gtimes))

