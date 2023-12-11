import pandas as pd
import numpy as np
import csv
import xlwt
import random
datalist=[0]*24
supply = pd.read_csv('pro3.csv')
supply=supply.values
supply=np.matrix(supply)
for i in range(supply.shape[0]):
    for j in range(2,26):
        datalist[j-2]+=supply[i,j]

for i in range(1,len(datalist)):
    datalist[i]+=datalist[i-1]
print(datalist)
for x in range(28000,40000):
    valid=True
    if datalist[0]<2*x:
        valid=False
    for j in range(1,len(datalist)):
        if datalist[j]<(j+2)*x:
            valid=False
    if valid==False:
        break
    else:
        print(x)
            
    



