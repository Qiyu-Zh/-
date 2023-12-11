import pandas as pd
import numpy as np
import csv
import xlwt
import random
supply = pd.read_csv('supply.csv')
demand= pd.read_csv('demand.csv')
demand=demand.values
demand=np.matrix(demand)
supply=supply.values
supply=np.matrix(supply)
for i in range(demand.shape[0]):
    d=demand[i,:]
    s=supply[i,:]

    for j in range(2,26):
        short=False
        m_s=0
        sum=0
        for k in range(5):
            if s[0,j+k*48]>m_s:
                m_s=s[0,j+k*48]
                sum+=s[0,j+k*48]  
            if s[0,j+k*48]<d[0,j+k*48]:
                short=True
        if short==False:
            supply[i,j]=m_s*random.uniform(1.1,1.5)
        else:
            
            supply[i,j]=random.uniform(sum/10,m_s)

f = xlwt.Workbook()  # 创建工作簿
sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
  # h为行数，l为列数
for i in range(demand.shape[0]):
    
    for j in range(26):
        if j>1:
            if supply[i, 1]=='A':
                sheet1.write(i, j, supply[i, j])
            if supply[i, 1]=='B':
                sheet1.write(i, j, supply[i, j])
            if supply[i, 1]=='C':
                sheet1.write(i, j, supply[i, j])
        else:
            sheet1.write(i, j, supply[i, j])

f.save('pro3.xlsx')
