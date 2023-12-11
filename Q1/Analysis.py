import pandas as pd
import csv
import numpy as np
np.set_printoptions(suppress=False)

# d = pd.read_csv('Data\Top50.csv')
# d = np.matrix(d.values)
# data = d[:,3:]

# mean = np.mean(data,axis = 1)
# print(np.shape(mean))
# var = np.var(data, axis = 1)
# mean_var = np.hstack ((mean, var))
# # print(mean_var)
# dataframe = pd.DataFrame(mean_var)
# dataframe.to_csv(r'Data/Top50_sta.csv')

supply = pd.read_csv('Data/1-供应商的供货量.csv')
demand = pd.read_csv('Data/1-企业的订货量.csv')
supply = supply.values
demand = demand.values
id = supply[:,0]
product = supply[:,1]
supply = np.matrix(supply[:,2:])
demand = np.matrix(demand[:,2:])
surplus = supply - demand

var = np.var(surplus, axis = 1)
print(var)