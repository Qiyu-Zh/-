#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# get_ipython().run_line_magic('matplotlib', 'inline')


# 数据预处理  
# 
# 将附件1两个sheet分成两张表格，分别命名为“订货量”与“供货量” 

# In[2]:


order = pd.read_csv(r'Data/1-企业的订货量.csv')
provide = pd.read_csv(r'Data/1-供应商的供货量.csv')


# In[3]:


columns = ['供应商ID','材料分类']+ [i for i in range(1, 241)]
order.columns=columns
provide.columns=columns


# In[4]:


order.head()


# In[5]:


provide.head()


# 供应量每周分布，简单选择5个看一下

# In[84]:


column = [i for i in range(1, 241)]
fenbu_pic = provide.head()[column]
fenbu_pic.index=['S00'+str(i) for i in range(1, 6)]
fenbu_pic=fenbu_pic.T
fenbu_pic['1']=fenbu_pic.index
ax, fig=plt.subplots( figsize=(30, 15))
for i in range(1,6):
    axi = ax.add_subplot(2, 3, i)
    axi.plot(fenbu_pic['1'],fenbu_pic['S00'+str(i)], label='S00'+str(i))
#     axi.title('S00'+str(i))
plt.savefig('S001-S005供应量每周分布')
plt.show()


# 每周订单量以及供应量

# In[134]:


plt.plot(provide[column].sum(),label='每周供应量')
plt.plot(order[column].sum(),label='每周订单量')
plt.title('每周数量')
plt.legend()


# 计算供应商的总订单数，均值以及方差

# In[7]:


provide['总计订单数']=np.sum(provide[columns], axis=1)
provide['供货量方差']=np.var(provide[columns], axis=1)
provide['供货量均值']=np.mean(provide[columns], axis=1)
provide


# 绘制三种原料供应比例变化图

# In[8]:


kind_num = copy.deepcopy(provide[1:4][column])
kind_num.index=['A','B','C']
kind_num


# In[9]:

provide_A = np.sum(provide[provide['材料分类']=='A'][column])
provide_B = np.sum(provide[provide['材料分类']=='B'][column])
provide_C = np.sum(provide[provide['材料分类']=='C'][column])
kind_num.loc['A',:]=provide_A.T
kind_num.loc['B',:]=provide_B.T
kind_num.loc['C',:]=provide_C.T
kind_provide=kind_num.copy()
for i in kind_num.columns:
    sum_num = kind_num.loc[:,i].sum()
    kind_num.loc['A',i]/=sum_num
    kind_num.loc['B',i]/=sum_num
    kind_num.loc['C',i]/=sum_num
kind_num=kind_num.T
kind_num


# In[83]:


plt.bar(kind_num.index, kind_num['A']*100,label='A')
plt.bar(kind_num.index, kind_num['B']*100,label='B')
plt.bar(kind_num.index, kind_num['C']*100,label='C')
plt.legend(['A','B','C'])
plt.title('三种原材料每周供应比例对比')
plt.savefig('三种原材料每周供应比例对比')
plt.show()


# In[82]:


plt.savefig('三种原材料每周供应比例对比')


# 绘制三种原料订购比例变化图

# In[11]:


kind_order = copy.deepcopy(order[1:4][column])
kind_order.index=['A','B','C']
kind_order


# In[12]:


order_A = np.sum(order[order['材料分类']=='A'][column])
order_B = np.sum(order[order['材料分类']=='B'][column])
order_C = np.sum(order[order['材料分类']=='C'][column])
kind_order.loc['A',:]=order_A.T
kind_order.loc['B',:]=order_B.T
kind_order.loc['C',:]=order_C.T
# print(kind_num)
kind_order_num=kind_order
for i in kind_order.columns:
    sum_num = kind_order.loc[:,i].sum()
    kind_order.loc['A',i]/=sum_num
    kind_order.loc['B',i]/=sum_num
    kind_order.loc['C',i]/=sum_num
kind_order=kind_order.T
kind_order


# In[13]:


plt.bar(kind_order.index, kind_order['A']*100,label='A')
plt.bar(kind_order.index, kind_order['B']*100,label='B')
plt.bar(kind_order.index, kind_order['C']*100,label='C')
plt.legend(['A','B','C'])
plt.title('三种原材料每周订购比例对比')
plt.savefig('三种原材料每周订购比例对比')
plt.show()


# 供应商订货量每周占比

# In[14]:


provide_pro = provide[['供应商ID','材料分类']+column].copy()
provide_pro
for i in provide_pro.index:
    provide_pro.loc[i, column]/=kind_provide.loc[provide_pro.loc[i, '材料分类'],:]
provide_pro  


# 供货率=供货量/订货量  
# 满货率=供货率>1的次数/总次数  
# 进阶特征量化评估：  
# 可以使用所有的供订货量训练，最后预测订100的货，能到多少
# 

# In[15]:


sat_pro = provide[['供应商ID','材料分类']+column].copy()
sat_pro.head()


# In[16]:


order.head()[column]


# In[17]:


for i in sat_pro.index:
    (sat_pro.loc[i, column])/=(order.loc[i, column])
sat_pro


# 对于上表的nan表示的是没有进行预订，虽然可以通过+1的方式计算出满货率为100%，但是如果本身订货次数较低，最后的总的满货率却会偏高

# In[18]:


sat_pro['订货次数']=0
for i in sat_pro.index:
    sat_pro.loc[i, '订货次数']=sat_pro.loc[i, :].count()-2
sat_pro


# In[19]:


sat_pro_na=sat_pro.fillna(0).copy()
sat_pro_na


# In[20]:


sat_pro['满货率']=0
for i in sat_pro.index:
    sat_pro_na.loc[i, '满货率']=sat_pro_na.loc[i, column].sum()/sat_pro_na.loc[i, '订货次数']
sat_pro_na


# In[ ]:





# In[88]:


provide['满货率']=sat_pro_na['满货率']
provide['订货次数']=sat_pro_na['订货次数']
provide['供货量有效均值']=provide['供货量均值']*240/provide['订货次数']
provide['最大供货量']=np.max(provide[column],axis=1)
provide['供货量']=provide['最大供货量']/3+provide['供货量有效均值']*2/3
provide


# In[22]:


sat_pro_na.sort_values(by='满货率',ascending=False)


# 评分的简单算法，可以直接的加权求和   
# provide\['评分'\]=a\*provide\['满货率'\]+b\*provide\['供货量方差'\]+c\*provide\['供货量均值'\]+d\*provide\['总计订单数'\]

# # 第二问  
# 这里简单是用满货率选择前50个供应商

# In[ ]:





# In[90]:


well_provider=provide.sort_values(by='满货率',ascending=False).head(50)
# well_provider['最大供货量']=np.max(well_provider[column],axis=1)
well_provider


# In[ ]:





# In[91]:


provider_info=well_provider[['供应商ID','材料分类','供货量有效均值','满货率','最大供货量','供货量']].copy()
provider_info.index=[i for i in range(0, 50)]
provider_info


# 这里的供应商的供货量=最大供货量/3+供货量有效均值\*2/3，如果只使用均值不能满足条件，可以自己选择中位数或者其他计算过的数作为供应商的供货参考量  
# 选择最少的供货商，达到两周生产量的需求：  
# 0-1规划，这里的xi中的i对应上表的索引+1  
# lingo程序求解(1)  
# ```lingo
# min=x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16+x17+x18+x19+x20+x21+x22+x23+x24+x25+x26+x27+x28+x29+x30+x31+x32+x33+x34+x35+x36+x37+x38+x39+x40+x41+x42+x43+x44+x45+x46+x47+x48+x49+x50;
# A/0.6+B/0.66+C/0.72>=2.82*10000;
# A=x1*7+x4*6+x6*5+x15*5+x21*12+x23*71+x26*4+x27*12+x28*4+x34*3430+x37*3+x41*5+x42*31+x44*46+x45*3+x47*5;
# B=x2*5+x5*5+x7*5+x8*5+x10*6+x11*6+x13*5+x16*5+x20*4+x24*6+x29*792+x30*4+x32*6+x33*3852+x35*5+x36*6+x38*4+x39*5+x43*8009+x48*6;
# C=x3*3+x9*5+x12*2+x14*8034+x17*3+x18*6+x19*7+x22*9+x25*2+x31*6+x40*5+x46*479+x49*23+x50*18;
# @bin(x1);@bin(x2);@bin(x3);@bin(x4);@bin(x5);
# @bin(x6);@bin(x7);@bin(x8);@bin(x9);@bin(x10);
# @bin(x11);@bin(x12);@bin(x13);@bin(x14);@bin(x15);
# @bin(x16);@bin(x17);@bin(x18);@bin(x19);@bin(x20);
# @bin(x21);@bin(x22);@bin(x23);@bin(x24);@bin(x25);
# @bin(x26);@bin(x27);@bin(x28);@bin(x29);@bin(x30);
# @bin(x31);@bin(x32);@bin(x33);@bin(x34);@bin(x35);
# @bin(x36);@bin(x37);@bin(x38);@bin(x39);@bin(x40);
# @bin(x41);@bin(x42);@bin(x43);@bin(x44);@bin(x45);
# @bin(x46);@bin(x47);@bin(x48);@bin(x49);@bin(x50);
# ```
# 使用供应量作为参考，选择x14,x34,x43即可满足

# 在(1)的基础上，去掉了最少供应商的要求，增加了采购费用的限制  
# lingo程序求解(2)
# ```lingo
# min=1.2*A+1.1*B+C;
# A/0.6+B/0.66+C/0.72>=2.82*10000;
# A=x1*7+x4*6+x6*5+x15*5+x21*12+x23*71+x26*4+x27*12+x28*4+x34*3430+x37*3+x41*5+x42*31+x44*46+x45*3+x47*5;
# B=x2*5+x5*5+x7*5+x8*5+x10*6+x11*6+x13*5+x16*5+x20*4+x24*6+x29*792+x30*4+x32*6+x33*3852+x35*5+x36*6+x38*4+x39*5+x43*8009+x48*6;
# C=x3*3+x9*5+x12*2+x14*8034+x17*3+x18*6+x19*7+x22*9+x25*2+x31*6+x40*5+x46*479+x49*23+x50*18;
# @bin(x1);@bin(x2);@bin(x3);@bin(x4);@bin(x5);
# @bin(x6);@bin(x7);@bin(x8);@bin(x9);@bin(x10);
# @bin(x11);@bin(x12);@bin(x13);@bin(x14);@bin(x15);
# @bin(x16);@bin(x17);@bin(x18);@bin(x19);@bin(x20);
# @bin(x21);@bin(x22);@bin(x23);@bin(x24);@bin(x25);
# @bin(x26);@bin(x27);@bin(x28);@bin(x29);@bin(x30);
# @bin(x31);@bin(x32);@bin(x33);@bin(x34);@bin(x35);
# @bin(x36);@bin(x37);@bin(x38);@bin(x39);@bin(x40);
# @bin(x41);@bin(x42);@bin(x43);@bin(x44);@bin(x45);
# @bin(x46);@bin(x47);@bin(x48);@bin(x49);@bin(x50);
# ```
# 使用供应量作为参考，选择x14,x34,x43即可满足

# In[97]:


# lingo代码生成器，x1*数字的那一段
# for i in provider_info.index:
#     print('x'+str(i+1)+'*'+'%.0f'%(provider_info['供货量'][i]), end='+')
# for i in provider_info.index:
#     print('@bin('+'x'+str(i+1)+');')
for i in provider_info.index:
    if provider_info['材料分类'][i]=='C':
        print('x'+str(i+1)+'*'+'%.0f'%(provider_info['供货量'][i]), end='+')


# In[26]:


trans = pd.read_excel(r'附件2 近5年8家转运商的相关数据.xlsx')
# trans.columns=[['转运商ID']+[i+1 for i in range(240)]]
# # trans.index=[i for i in range(8)]
trans


# # 附件2分析
# 对转运表的处理：  
# 因为不知道转运商具体转运的数量，以及货物的类型，所以我们假设转运表中的所有数据不相关   
# 此外，因为不知道转运商与供货商以及仓库的位置，所以假设转运商的区别只在于损耗率，其他一切相同  
# 所以使用均值，方差以及中位数作为转运商的评定标准以及5年的接收到订单的次数  
# 对于均值、方差以及中位数的计算都不考虑0的情况，只考虑接收到订单的情况

# In[36]:


a = ['W00'+str(i) for i in range(1, 9)]
a+=['W0'+str(i) for i in range(10, 99)]
a+=['W'+str(i) for i in range(100, 241)]
trans['订单次数']=(trans[a]>0.0000001).sum(axis=1)
trans['均值']=trans[a].mean(axis=1)*240/trans['订单次数']
trans['方差']=trans[a].var(axis=1)
trans


# 这里直接以均值和订单占比的盘平均数作为损耗率，也可以参考其他参数加权后得出新的损耗率   
# 从这里看，转运商T3的效果最好，T6次之，所以安排转运商的时候，先把T3的每周6000安排满，再安排T6，以此类推

# In[86]:


tans_info = trans[['转运商ID','订单次数','均值','方差']].copy()
tans_info['订单占比']=tans_info['订单次数']/240
tans_info['评分']=(tans_info['订单占比']+tans_info['均值'])/2
tans_info


# In[85]:


fig=plt.figure()
ax = fig.add_subplot(111, polar=True)
for i in range(8):
    values = list(tans_info.loc[i, ['均值','方差','订单占比']])

    feature = ['订单占比','均值','方差','订单占比']
    angles=np.linspace(0, 2*np.pi,len(values), endpoint=False)
    values=np.concatenate((values,[values[0]]))
    angles=np.concatenate((angles,[angles[0]]))

    
    ax.plot(angles, values, 'o-', linewidth=2,label=tans_info['转运商ID'][i])
    ax.fill(angles, values, alpha=0.25)
    # ax.fill(angles, values_2, alpha=0.25)

    ax.set_thetagrids(angles * 180/np.pi, feature)
plt.legend(loc='best')
ax.grid(True)
plt.savefig('均值-方差-订单占比雷达图')
plt.show()


# In[87]:


tans_info.sort_values(by='评分')


# # 问题2订单量表格

# （设第一周没下单之前仓库里刚好有两周的货,实际上有具体的数值可以通过供货量计算出来）   
# 所以第一周按照产能需求进货，第二周之后根据仓库里的库存适当调整进货量。
# 对于供应商的选择：  
# 只选择满货率前50的供应商，对于上周已经选择过的供应商，两周之内暂不考虑该供应商（即默认供应商两周可以达到最大供货量，更详细的生产量可以使用机器学习去分析）  
# 关于供应商实际的供应量：  
# 为订单量与最大供应量之间的随机整数

# In[127]:


order_week = order[['供应商ID']].copy()
order_week.index=order['供应商ID'].values
order_week


# In[126]:


# for i in range(25):
#     order_week['%d'%i]=0
#     # （英文逗号连接，只输入角标就行如x31输入31）
#     ans=input('输入lingo的求解结果：').split(',')
#     for j in ans:
#         order_week['%d'%i][provider_info['供应商ID'][j]]=np.random.randint(provider_info['供货量'][j],provider_info['最大供货量'][j])
#     need=sum(order_week['%d'%i])
#     print('min=1.2*A+1.1*B+C;\n')

