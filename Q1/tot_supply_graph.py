import pandas as pd
import csv
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=False)
plt.rcParams['font.sans-serif']=['SimHei']

supply = pd.read_csv('Data/1-供应商的供货量.csv')
demand = pd.read_csv('Data/1-企业的订货量.csv')
supply = supply.values[1:,2:]
demand = demand.values

tot_supply = np.sum(supply, axis=1)
argsort = np.argsort(tot_supply)[::-1]
sorted_tot = tot_supply[argsort]
print(sorted_tot)
x = np.array([i for i in range(1,402)])
plt.plot(x,sorted_tot)
plt.xlabel(r'供货量排名')
plt.ylabel(r'总供货量')
plt.show()

# gap = supply - demand

# # print(np.shape(supply))
# # tot_supply = np.sum(supply, axis = 1)

# # indexes = np.argsort(tot_supply,axis=0)[:10]
# # print(indexes)
# # supply1 = supply#[:,120:]
# # # print(supply1)
# # print(np.shape(supply1))
# # fig, ax = plt.subplots()
# # for ind in indexes:
# #     ax.plot(range(121,241),np.reshape(supply1[ind,:],(120,-1)))

# supply = supp[:,1:]
# supp_A = supply[supply['材料分类']=='A']
# print(supp_A)

plt.show()