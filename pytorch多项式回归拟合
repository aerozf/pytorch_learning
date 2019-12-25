# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 11:17:52 2019

@author: zhangfan
"""

import torch,pylab
import numpy as np

learning_rate=1.0e-7  #很重要的一个参数

def make_features(x):
    x=x.unsqueeze(1)
    return torch.cat([x**i for i in range(1,4)],1)


W_get=torch.FloatTensor([1,1,1]).unsqueeze(1)
b_get=torch.FloatTensor([0.9])


def f(x):
    return x.mm(W_get)+b_get[0]

def get_batch(bsize=100):
    x=np.linspace(-10,10,bsize)
    x=make_features(torch.FloatTensor(x))
    y=f(x)+torch.FloatTensor(np.random.random(bsize)*20).unsqueeze(1)
    return torch.autograd.Variable(x),torch.autograd.Variable(y)
#建立模型
class Poly(torch.nn.Module):
    def __init__(self):
        super(Poly,self).__init__()
        self.poly=torch.nn.Linear(3,1) #输入输出均为1维
        
    def forward(self,x):
        out=self.poly(x)
        return out
  

model=Poly()

#优化
criterion=torch.nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#开始训练
num=5000

batch_x,batch_y=get_batch()

pylab.ion()
pylab.figure(1)
px=[]
py=[]
for e in range(num):
    
    #获取数据
    #forward
    out=model(batch_x)
    loss=criterion(out.float(),batch_y.float())
    
    ploss=loss.data
    px.append(e+1)
    py.append(ploss)
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
 

    
x=np.linspace(-10,10,32)
x=make_features(torch.FloatTensor(x))
pylab.figure(2)     
model.eval()
predict=model(torch.autograd.Variable(x.float()))
#print(model(torch.autograd.Variable(torch.Tensor([0,0,0]))))  #x=0时的预测值
pylab.plot(batch_x.numpy()[:,0],batch_y.numpy(),'bx',label='Original data') #[:,0为换取矩阵第一列所有元素]
pylab.plot(x.numpy()[:,0],predict.data.numpy(),'r')
pylab.pause(0)
