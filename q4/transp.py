
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
# def cost_A(value):
#     if value<=6000:
#         value*=0.99814
#         return value[]
#     if 12000>=value>6000:
#         value=6000*0.99814+(value-6000)*0.99456
#     if 18000>=value>12000:
#         value=6000*0.99814+6000*0.99456+(value-12000)*0.99080

supply = pd.read_csv('pro2.csv')
supply=supply.values
supply=np.matrix(supply)
trans=[[6000]*24 for i in range(8)]
trans[1][5]=0
trans[3][16]=0

f = xlwt.Workbook()  # 创建工作簿
sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
for j in range(0,supply.shape[1]):
    for i in range(supply.shape[0]):
        if supply[i,j]==0:
            continue
        for k in range(8):
            print(type(supply[i,j]),i,j)
            if supply[i,j]<=trans[k][j]:
                sheet1.write(i, j*8+pos(k), supply[i,j])
                trans[k][j]-=supply[i,j]
                break
            if supply[i,j]>trans[k][j]:
                if trans[k][j]!=0:
                    sheet1.write(i, j*8+pos(k), trans[k][j])
                supply[i,j]-=trans[k][j]
                trans[k][j]=0

f.save('tranP2_ans.xlsx')