from datacopy import distribute
import pandas as pd
import numpy as np
import csv
import xlwt
def pos(value):
    if value==0:
        return 2
    if value==1:
        return 5
    if value==2:
        return 1
    if value==3:
        return 7
    if value==4:
        return 3
    if value==5:
        return 0
    if value==6:
        return 6
    if value==7:
        return 4

supply = pd.read_csv('pro3_q1Fi.csv',header=None)
print(supply.shape)
supply=supply.values
supply=np.matrix(supply)
for i in range(supply.shape[0]):
    for j in range(supply.shape[1]):
        if supply[i,j]==np.nan:
            supply[i,j]=0
trans=[[6000]*24 for i in range(8)]
trans[1][5]=0
trans[3][16]=0

f = xlwt.Workbook()  # 创建工作簿
sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
for j in range(24):
    for i in range(402):
        if supply[i,j]==0:
            continue
        for k in range(8):
            print(i,j)
            if supply[i,j]<=trans[k][j]:
                sheet1.write(i, j*8+pos(k), supply[i,j])
                trans[k][j]-=supply[i,j]
                break
            if supply[i,j]>trans[k][j]:
                if trans[k][j]!=0:
                    sheet1.write(i, j*8+pos(k), trans[k][j])
                supply[i,j]-=trans[k][j]
                trans[k][j]=0

f.save('tran_ans.xlsx')