#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from gurobipy import *
import json
import re


# ## define

# In[2]:


W=8         # the width of NoC
H=8         # the hieght of Noc
S=W*H       # the number of core
C=1024      # capacity of a core
x=[]        # the horizontal coordinate of core_i
y=[]        # the vertical coordinate of core_j
for i in range(W):
    for j in range(H):
        x.append(i)
        y.append(j)


# ## load data

# In[3]:


#sys.argv.append('-t')
#sys.argv.append('36000')
time_limit = 0
arg = sys.argv
for i in range(len(arg)):
    if arg[i]=='-t':
        time_limit = int(arg[i+1])
        #print(time_limit)
file = open("info.txt", encoding = 'utf-8')
text = json.load(file)
file.close()

core2num = {}
num2core = {}
axon2core = {}
neuron2core = {}
N = len(text['cores'])
print(N)
d=[[0]*N for i in range(N)]
axon_size = [0]*N
neuron_size = [0]*N
l=[0]*N
N = 0

for core in text['cores']:
    core_name = core['core_name']
    core2num[core_name] = N
    num2core[N] = core_name
    axon_size[N] = core['axon_num']
    neuron_size[N] = core['neuron_num']
    l[N] = core['neuron_num']
    N += 1
    for axon in core['axons']:
        axon2core[axon['axon_name']] = core_name

#print(type(core2num),type(num2core),type(axon2core),type(neuron2core))

for core in text['cores']:
    for neuron in core['neurons']:
        neuron2core[neuron['neuron_name']] = core['core_name']
        xi = core2num[core['core_name']]
        #print(neuron['route_to'])
        #print(neuron['route_to'],axon2core[neuron['route_to']],'!')
        if neuron['route_to']=='C-1A-1':
            continue
        yi = core2num[axon2core[neuron['route_to']]]
        d[xi][yi] += 1

for i in range(N):
    for j in range(N):
        if d[i][j]>0 and d[j][i]>0:
            print(i,j)
# print(axon_size)
# print(neuron_size)
# print(l)


# ## ILP

# In[4]:


try:
    #create a new model
    m=Model("vgg Mapping")
    
    m.setParam('MIPGap',0.1)
    if time_limit != 0:
        m.setParam('TimeLimit',time_limit)
    m.setParam('NumericFocus',1)
   # m.setParam('MIPFocus',3)    

    # create variables
    a=m.addVars(N,S,vtype=GRB.BINARY,name="a")  # if neuron_i is mapped into core_j
    f=m.addVars(N,vtype=GRB.INTEGER,name="f")   # the time when neuroni is finished
    b=m.addVars(N,N,vtype=GRB.BINARY,name="b")  # auxiliary variables for the serial core constraint
    t=m.addVars(N,N,vtype=GRB.INTEGER,name="t") # data transfer latency from neuron_i to neuron_j
    ans=m.addVar(vtype=GRB.INTEGER,name="ans")      # maximum f_i
    energy=m.addVar(vtype=GRB.INTEGER,name="energy")

    # set objective
    m.setObjective(ans,GRB.MINIMIZE)

    # add constraint
    m.addConstr(ans==max_(f[i] for i in range(N)),name="ans=maximum f_i")
    m.addConstr(energy==t.sum('*','*'))
    m.addConstrs((f[i]>=512*l[i] for i in range(N)),name="finish time is none-negative")
    m.addConstrs((t[i,j]>=0 for i in range(N) for j in range(N)),
                 name="transfer latency is none-negative")
    
    #core capacity constraint
#     for i in range(N):
#         m.addConstr(a[i,i]==1)
    for j in range(S):
        exp=LinExpr(0)
        for i in range(N):
            exp+=axon_size[i]*a[i,j]
        m.addConstr(exp<=C)
    for j in range(S):
        exp=LinExpr(0)
        for i in range(N):
            exp+=neuron_size[i]*a[i,j]
        m.addConstr(exp<=C)
    m.addConstrs((a.sum(i,'*')==1 for i in range(N)),name="no copy constraint")
    for j in range(N):
        degree = 0
        for i in range(N):
            degree += d[i][j]
        if degree == 0 :
            #print(j)
            m.addConstr(a[j,0]+a[j,8]+a[j,16]+a[j,24]+a[j,32]+a[j,40]+a[j,48]+a[j,56]==1)

    m.addConstrs((f[i]+t[i,j]+512*l[i]<=f[j] for i in range(N) for j in range(N) if d[i][j]>0),
                 name = "neuron order constraint")
    m.addConstrs((-100000000000*b[i,j]-1000000000000*(1-a[i,k])-1000000000000*(1-a[j,k])<=f[i]-l[i]-f[j] 
                  for i in range(N) for j in range(i) for k in range(S)),
                 name = "serial core constraint1")
    m.addConstrs((100000000000*(b[i,j]-1)-1000000000000*(1-a[i,k])-1000000000000*(1-a[j,k])<=f[j]-l[i]-f[i] 
                  for i in range(N) for j in range(i) for k in range(S)),
                 name = "serial core constraint2")
    for i in range(N):
        for j in range(N):
            if d[i][j]<=0:
                m.addConstr(t[i,j]==0)
            expr1=LinExpr(0)
            expr2=LinExpr(0)
            expr3=LinExpr(0)
            expr4=LinExpr(0)
            for k in range(S):
                expr1+=x[k]*a[i,k]-x[k]*a[j,k]+y[k]*a[i,k]-y[k]*a[j,k]
                expr2+=x[k]*a[j,k]-x[k]*a[i,k]+y[k]*a[i,k]-y[k]*a[j,k]
                expr3+=x[k]*a[i,k]-x[k]*a[j,k]+y[k]*a[j,k]-y[k]*a[i,k]
                expr4+=x[k]*a[j,k]-x[k]*a[i,k]+y[k]*a[j,k]-y[k]*a[i,k]
            m.addConstr(d[i][j]*expr1<=t[i,j])
            m.addConstr(d[i][j]*expr2<=t[i,j])
            m.addConstr(d[i][j]*expr3<=t[i,j])
            m.addConstr(d[i][j]*expr4<=t[i,j])
            #m.addConstr(t[i,j]<=f[i][j])   
    m.optimize()
    
    map = {}    #core2ans
    for i in range(N):
        for j in range(S):
            #var=m.getVarByName("a[i,j]")
            if a[i,j].getAttr("x")>0.01:
                map[num2core[i]] = [j//W, j%H]
    
except GurobiError as e:
    print(e.errno)
    print(e.message)


# ## output

# In[5]:


def getID(s):
    List = re.findall(r'\d+',s)
    return List[len(List)-1]


# In[6]:


neuron_dict = {}
axon_dict = {}
core_dict = {}
for item in neuron2core.items():
    neuron_dict[item[0]] = [[0,0], map[item[1]], getID(item[0])]
for item in axon2core.items():
    axon_dict[item[0]] = [[0,0], map[item[1]], getID(item[0])]
for item in num2core.items():
    core_dict[item[1]] = [[0,0], map[item[1]]]

json_dict = {}
json_dict['neurons']=neuron_dict
json_dict['axons']=axon_dict
json_dict['cores']=core_dict
with open('map_result.txt','w') as f:
    json.dump(json_dict, f)


# In[7]:


f = open("utility.txt","w")
ans_str="core = "+str(len(core_dict))+" axon = "+str(len(axon_dict))+" neuron = "+str(len(neuron_dict))+"\n"
f.write(ans_str)
tot_axon=[[0]*8 for i in range(8)]
tot_neuron=[[0]*8 for i in range(8)]
for core in text['cores']:
    xi = core_dict[core['core_name']][1][0]
    yi = core_dict[core['core_name']][1][1]
    tot_axon[xi][yi] += core['axon_num']
    tot_neuron[xi][yi] += core['neuron_num']
use = 0
for xi in range(8):
    for yi in range(8):
        if tot_axon[xi][yi] != 0:
            use += 1
        ans_str="Core("+str(xi)+","+str(yi)+"): axon_utility="+str(tot_axon[xi][yi]/1024)+" neuron_utility="+str(tot_neuron[xi][yi]/1024)+"\n"
        f.write(ans_str)
f.close()
print(use)


# In[ ]:




