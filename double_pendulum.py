#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 20:46:25 2020

@author: zen
"""
import numpy as np
import matplotlib.pyplot as plt

import learner as ln
from learner.integrator.hamiltonian import SV

class DBData(ln.Data):
    '''Data for learning the double pendulum system with the Hamiltonian  H(p1,p2,q1,q2)
    = (m2l2^2p1^2 + (m1+m2)l_1^2p2^2 - 2m2l1l2p1p2cos(q1-q2))/(2m2l1^2l2^2(m1+m2sin^2(q1-q2)))
    -(m1+m2)gl1cosq1 - m2gl2cosq2.
    '''
    def __init__(self, x0, h, train_num, test_num, add_h=False):
        super(DBData, self).__init__()
        self.solver = SV(None, self.dH, iterations=10, order=6, N=5)
        self.x0 = x0
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        self.add_h = add_h
        self.__init_data()
        
    @property
    def dim(self):
        return 4
    
    def __generate_flow(self, x0, h, num):
        X = self.solver.flow(np.array(x0), h, num)
        x, y = X[:-1], X[1:]
        if self.add_h:
            x = np.hstack([x, self.h * np.ones([x.shape[0], 1])])
        return x, y
    
    def __init_data(self):
        self.X_train, self.y_train = self.__generate_flow(self.x0, self.h, self.train_num)
        self.X_test, self.y_test = self.__generate_flow(self.y_train[-1], self.h, self.test_num)
    
    def dH(self, p, q):
        p1 = p[..., 0]
        p2 = p[..., 1]
        q1 = q[..., 0]
        q2 = q[..., 1]
        h1 = p1*p2*np.sin(q1-q2)/(1+np.sin(q1-q2)**2)
        h2 = (p1**2+2*p2**2-2*p1*p2*np.cos(q1-q2))/2/(1+np.sin(q1-q2)**2)**2
        dHdp1 = (p1-p2*np.cos(q1-q2))/(1+np.sin(q1-q2)**2)
        dHdp2 = (-p1*np.cos(q1-q2)+2*p2)/(1+np.sin(q1-q2)**2)
        dHdq1 = 2*np.sin(q1)+h1-h2*np.sin(2*(q1-q2))
        dHdq2 = np.sin(q2)-h1+h2*np.sin(2*(q1-q2))
        dHdp = np.hstack([dHdp1, dHdp2])
        dHdq = np.hstack([dHdq1, dHdq2])
        return dHdp, dHdq
        
def plot(data, net):
    t_test = np.arange(0, data.h*data.test_num, data.h)
    if isinstance(net, ln.nn.HNN):
        flow_true = data.solver.flow(data.X_test_np[0][:-1], data.h, data.test_num-1)
        flow_pred = net.predict(data.X_test[0][:-1], data.h, data.test_num-1, keepinitx=True, returnnp=True)
    else:
        flow_true = data.solver.flow(data.X_test_np[0], data.h, data.test_num-1)
        flow_pred = net.predict(data.X_test[0], data.test_num-1, keepinitx=True, returnnp=True)
        
    plt.figure(figsize=[6 * 2, 4.8 * 1])    
    plt.subplot(121)
    plt.plot(t_test, flow_true[:, 2], color='b', label='Ground truth', zorder=0)
    plt.scatter(t_test, flow_pred[:, 2], color='r', label='Predicted solution', zorder=1)
    plt.ylim([-1.5,2])
    plt.title('Pendulum 1')
    plt.legend(loc='upper left')
    plt.subplot(122)
    plt.plot(t_test, flow_true[:, 3], color='b', label='Ground truth', zorder=0)
    plt.scatter(t_test, flow_pred[:, 3], color='r', label='Predicted solution', zorder=1)
    plt.ylim([-1.5,2])
    plt.title('Pendulum 2')
    plt.legend(loc='upper left')
    plt.savefig('double_pendulum.pdf')

def main():
    device = 'cpu' # 'cpu' or 'gpu'
    # data
    x0 = [0, 0, np.pi*3/7, np.pi*3/8]
    h = 0.75
    train_num = 200
    test_num = 100
    # net
    net_type = 'LA' # 'LA' or 'G' or 'HNN'
    LAlayers = 8
    LAsublayers = 5
    Glayers = 8
    Gwidth = 50
    activation = 'sigmoid'
    Hlayers = 4
    Hwidth = 50
    Hactivation = 'tanh'
    # training
    lr = 0.001
    iterations = 50000
    print_every = 1000
    
    add_h = True if net_type == 'HNN' else False
    criterion = None if net_type == 'HNN' else 'MSE'
    data = DBData(x0, h, train_num, test_num, add_h)
    if net_type == 'LA':
        net = ln.nn.LASympNet(data.dim, LAlayers, LAsublayers, activation)
    elif net_type == 'G':
        net = ln.nn.GSympNet(data.dim, Glayers, Gwidth, activation)
    elif net_type == 'HNN':
        net = ln.nn.HNN(data.dim, Hlayers, Hwidth, Hactivation)
    args = {
        'data': data,
        'net': net,
        'criterion': criterion,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': None,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    plot(data, ln.Brain.Best_model())
    
if __name__ == '__main__':
    main()