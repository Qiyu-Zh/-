import pandas as pd
import numpy as np
import csv
import xlwt
import random
supply = pd.read_csv('pro3.csv',header=None)
supply=supply.values
supply=np.matrix(supply)
order = pd.read_csv('order.csv',header=None)
order=order.values
order=np.matrix(order)
total=[[0]*24 for i in range(3)]
distribute=[[0]*24 for i in range(3)]

for j in range(2,26):
    for i in range(supply.shape[0]):
        if supply[i,1]=='A':
            total[0][j-2]+=supply[i,j]*0.985
        if supply[i,1]=='C':
            total[1][j-2]+=supply[i,j]*0.985
        if supply[i,1]=='B':
            total[2][j-2]+=supply[i,j]*0.985

target=[56400]+[28200]*23
week=[total[0][i]/0.6+total[1][i]/0.72+total[2][i]/0.66 for i in range(0,24)]
for i in range(23,-1,-1):
    if week[i]>=target[i]:
        week[i]=target[i]
    if week[i]<target[i]:
        target[i-1]+=target[i]-week[i]
target=week

for i in range(len(total[0])):
    if target[i]-total[0][i]/0.6>0:
        distribute[0][i]=total[0][i]
        target[i]=target[i]-total[0][i]/0.6
    else:
        distribute[0][i]=target[i]*0.6
        target[i]=0

for j in range(len(total[1])):
    if target[j]==0:
        continue
    if target[j]-total[1][j]/0.72>0:
        distribute[1][j]=total[1][j]
        target[j]=target[j]-total[1][j]/0.72
    else:
        distribute[1][j]=target[j]*0.72
        target[j]=0

for k in range(24):
    if target[k]==0:
        continue
    if target[k]-total[2][k]/0.66>0:
        distribute[2][k]=total[2][k]
        target[k]=target[k]-total[2][k]/0.66
    else:
        distribute[2][k]=target[k]*0.66
        target[k]=0
pos=[0]*24
value=[0]*24

A=[24]

C=[22,21,23,19,18,20,16,15,14,13,12,11,17]

B=[10,2,3,4,5,6,7,8,9,14,25]
print(distribute[2])
for j in B:
    sum=0
    for i in range(145,279):
        sum+=supply[i,j]
        if sum>distribute[2][j-2]:
            pos[j-2]=i
            value[j-2]=distribute[2][j-2]-pre
            break
        else:
            pre=sum

for j in C:
    sum=0
    for i in range(279,402):
        sum+=supply[i,j]
        if sum>distribute[1][j-2]:
            pos[j-2]=i
            value[j-2]=distribute[1][j-2]-pre
            
            break
        else:
            pre=sum

for j in A:
    sum=0
    for i in range(0,145):
        sum+=supply[i,j]
        if sum>distribute[0][j-2]:
            pos[j-2]=i
            value[j-2]=distribute[0][j-2]-pre
            
            break
        else:
            pre=sum

f = xlwt.Workbook()  # 创建工作簿
sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
  # h为行数，l为列数
for i in range(402):
    for j in B:
        j-=2
        if i<145 or i >278:
            sheet1.write(i, j, order[i,j+2])
        else:
            if i<pos[j]:
                sheet1.write(i, j, order[i,j+2])
            if i==pos[j]:
                sheet1.write(i, j, value[j])
    for j in C:
        j-=2
        if i<145:
            sheet1.write(i, j, order[i,j+2])
        else:
            if 278<i<pos[j]:
                sheet1.write(i, j, order[i,j+2])
            if i==pos[j]:
                sheet1.write(i, j, value[j])
    for j in A:
        j-=2
        if i<pos[j]:
            sheet1.write(i, j, order[i,j+2])
        if i==pos[j]:
            sheet1.write(i, j, value[j])

f.save('pro3_q1Fi.xlsx')
    
    
    