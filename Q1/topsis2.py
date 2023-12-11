import pandas as pd
import csv
import numpy as np
np.set_printoptions(suppress=False)

supply = pd.read_csv('Data/109供货.csv')
demand = pd.read_csv('Data/109订货.csv')
supply = supply.values
demand = demand.values

id = supply[:,0]
product = supply[:,1]
supply = supply[1:,2:]
demand = demand[1:,2:]
surplus = supply - demand

# Accomplishment
nonzeros = np.zeros(np.shape(demand))
finished = np.zeros(np.shape(demand))
for i in range(np.shape(demand)[0]):
    for j in range(np.shape(demand)[1]):
        if demand[i][j] != 0:
            nonzeros[i][j] =1
        if surplus[i][j] >0:
            finished[i][j] =1
finish = np.sum(finished, axis=1)
nonzeros = np.sum(nonzeros, axis=1)
finish_rate = finish / nonzeros
# print(finish_rate)

# Supply
tot_supply = np.sum(supply, axis=1)
# print(type(tot_supply))
print(tot_supply)

# Supply Error
err_rate = np.zeros(np.shape(demand))
for i in range(np.shape(demand)[0]):
    for j in range(np.shape(demand)[1]):
        try:
            err_rate[i,j] = abs(surplus[i,j]/demand[i,j]) #只有这里加了绝对值
        except:
            pass
err_average = np.sum(err_rate, axis = 1) / np.count_nonzero(err_rate, axis=1)
err_average = np.max(err_average) - err_average #正向化

# 大额订单比例
price = demand
nonzeros = np.zeros(np.shape(demand))
bigs = np.zeros(np.shape(demand))
for i in range(np.shape(demand)[0]):
    for j in range(np.shape(demand)[1]):
        if demand[i][j] != 0:
            nonzeros[i][j] =1
        if supply[i][j] > 500:
            bigs[i][j] =1
big_tot = np.sum(bigs, axis=1)
num_purchase = np.sum(nonzeros)
big_rate = big_tot / num_purchase
# print(type(big_rate))
# print(np.shape(big_rate))

def z_stand(data):
    mean = data - np.mean(data)
    n = np.shape(data)[0]
    std = np.sqrt(np.var(data))*np.sqrt(n/(n-1))
    new = mean / std
    return new
def stand(data):
    min = np.min(data)
    max = np.max(data)
    return (data-min)/(max - min)
tot_supply = np.matrix(stand(tot_supply)).T
finish_rate = np.matrix(stand(finish_rate)).T 
err_average = np.matrix(stand(err_average)).T 
big_rate = np.matrix(stand(big_rate)).T 
# zero1 = stand(data[:,3])
# print(np.shape(demand1))
data = np.hstack((finish_rate, err_average, big_rate, tot_supply))
# print(data)
print(np.shape(data))

#余弦标准化
ss = np.sqrt(np.sum(np.square(data),axis = 0).astype('float'))
Z = data / ss
data = Z

maxs = np.max(data, axis = 0)
mins = np.min(data, axis = 0)
D1 = np.sqrt(np.sum(np.square(data-maxs), axis=1).astype('float'))
D0 = np.sqrt(np.sum(np.square(data-mins), axis=1).astype('float'))
# print(D1,D0)
score = np.array(D0 / (D1 + D0))
# print(score)
indexes = np.argsort(score, axis = 0)[-51:][::-1]
result = id[indexes]
print(id)
print(indexes)
# print(result)
dataframe = pd.DataFrame(result)
dataframe.to_csv(r'Data/Result1(1).csv')