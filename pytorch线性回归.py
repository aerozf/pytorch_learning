# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 11:17:52 2019

@author: zhangfan
"""

import torch,pylab
import numpy as np

x_t=np.linspace(1,10,20)
y_t=np.linspace(5,30,20)+np.random.random([1,20])*5
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

#开始训练y
pylab.ion()
pylab.figure(1)
px=[]
py=[]
num=1000
for e in range(num):
    inputs=torch.autograd.Variable(x_t)
    target=torch.autograd.Variable(y_t)
    
    #forward
    out=model(inputs.float())
    loss=criterion(out,target.float())
    ploss=loss.data
    px.append(e+1)
    py.append(ploss)
    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e+1)%500==0:
        #print('Epoch[{}/{}],loss:{:.6f}'.format(e+1,num,ploss))
        pylab.plot(px,py,'r.')
        pylab.title(str(ploss.data.numpy()))
        #pylab.draw()#注意此函数需要调用
        pylab.pause(0.01)         # 暂停一秒
        pylab.clf()
pylab.plot(px,py,'r.')
pylab.title('Loss: '+str(ploss.data.numpy()))
pylab.xlabel('iteration num')
pylab.ylabel("loss")
 
#预测值
x=np.linspace(-5,15,20)
x=torch.from_numpy(x)
x=x.view(20,1)  
model.eval()
predict=model(torch.autograd.Variable(x.float()))

#绘图
pylab.figure(2)
#print(model(torch.autograd.Variable(torch.Tensor([0,0,0]))))  #x=0时的预测值
pylab.plot(x_t.numpy()[:,0],y_t.numpy(),'b.',label='Original data') #[:,0为换取矩阵第一列所有元素]
pylab.plot(x.numpy()[:,0],predict.data.numpy(),'r',label='fitting data')
pylab.title('Pytorch Fitting')
pylab.xlabel('x value')
pylab.ylabel("y value")
pylab.legend()
pylab.pause(0)
