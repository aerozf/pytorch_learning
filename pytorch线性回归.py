# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 11:17:52 2019

@author: zhangfan
"""

import torch,pylab
import numpy as np

x_t=np.linspace(1,10,20)
y_t=np.linspace(5,30,20)+np.random.random([1,20])*10
y=y_t

#pylab.scatter(x_t,y_t)

x_t=torch.from_numpy(x_t).view(20,1)
y_t=torch.from_numpy(y_t).view(20,1)



#建立模型
class LinearReg(torch.nn.Module):
    def __init__(self):
        super(LinearReg,self).__init__()
        self.linear=torch.nn.Linear(1,1) #输入输出均为1维
        
    def forward(self,x):
        out=self.linear(x)
        return out
  

model=LinearReg()

#优化
criterion=torch.nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

#开始训练
num=10000
for e in range(num):
    inputs=torch.autograd.Variable(x_t)
    target=torch.autograd.Variable(y_t)
    
    #forward
    out=model(inputs.float())
    loss=criterion(out,target.float())
    
    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e+1)%100==0:
        print('Epoch[{}/{}],loss:{:.6f}'.format(e+1,num,loss.data))
        
        
model.eval()
predict=model(torch.autograd.Variable(x_t.float()))
pylab.plot(x_t.numpy(),y_t.numpy(),'r.',label='Original data')
pylab.plot(x_t.numpy(),predict.data.numpy())
