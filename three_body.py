#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:31:04 2020

@author: zen
"""

import numpy as np
import matplotlib.pyplot as plt

import learner as ln
from learner.integrator.hamiltonian import SV
np.random.seed(0)

class TBData(ln.Data):
    '''Data for learning the three body system.
    '''
    def __init__(self, h, train_traj_num, test_traj_num, train_num, test_num, add_h=False):
        super(TBData, self).__init__()
        self.solver = SV(None, self.dH, iterations=1, order=6, N=1)
        self.h = h
        self.train_traj_num = train_traj_num
        self.test_traj_num = test_traj_num
        self.train_num = train_num
        self.test_num = test_num
        self.add_h = add_h
        self.__init_data()
        
    @property
    def dim(self):
        return 12
    
    def dH(self, p, q):
        p1 = p[..., :2]
        p2 = p[..., 2:4]
        p3 = p[..., 4:6]
        q1 = q[..., :2]
        q2 = q[..., 2:4]
        q3 = q[..., 4:6]
        dHdp1 = p1
        dHdp2 = p2
        dHdp3 = p3
        dHdq1 = (q1-q2)/np.sum((q1-q2)**2, axis = -1, keepdims = True)**1.5 + (q1-q3)/np.sum((q1-q3)**2, axis = -1, keepdims = True)**1.5
        dHdq2 = (q2-q3)/np.sum((q2-q3)**2, axis = -1, keepdims = True)**1.5 + (q2-q1)/np.sum((q2-q1)**2, axis = -1, keepdims = True)**1.5
        dHdq3 = (q3-q1)/np.sum((q3-q1)**2, axis = -1, keepdims = True)**1.5 + (q3-q2)/np.sum((q3-q2)**2, axis = -1, keepdims = True)**1.5
        dHdp = np.hstack([dHdp1, dHdp2, dHdp3])
        dHdq = np.hstack([dHdq1, dHdq2, dHdq3])
        return dHdp, dHdq   
    
    def __generate_flow(self, x0, h, num):
        X = self.solver.flow(np.array(x0), h, num)
        x, y = X[:,:-1], X[:,1:]
        if self.add_h:
            x = np.concatenate([x, self.h * np.ones([x.shape[0], x.shape[1], 1])], axis = 2)
        return x, y
    
    def rotate2d(self, p, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s],[s, c]])
        R = np.transpose(R)
        return p.dot(R)
    
    def random_config(self, n, nu=2e-1, min_radius=0.9, max_radius=1.2, return_tensors=True):
        
        q1 = np.zeros([n, 2])
    
        q1 = 2*np.random.rand(n, 2) - 1
        r = np.random.rand(n) * (max_radius-min_radius) + min_radius
    
        ratio = r/np.sqrt(np.sum((q1**2), axis=1))
        q1 *= np.tile(np.expand_dims(ratio, 1), (1, 2))
        q2 = self.rotate2d(q1, theta=2*np.pi/3)
        q3 = self.rotate2d(q2, theta=2*np.pi/3)
    
        # # velocity that yields a circular orbit
        v1 = self.rotate2d(q1, theta=np.pi/2)
        v1 = v1 / np.tile(np.expand_dims(r**1.5, axis=1), (1, 2))
        v1 = v1 * np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2)) # scale factor to get circular trajectories
        v2 = self.rotate2d(v1, theta=2*np.pi/3)
        v3 = self.rotate2d(v2, theta=2*np.pi/3)
    
        # make the circular orbits slightly chaotic
        v1 *= 1 + nu*(2*np.random.rand(2) - 1)
        v2 *= 1 + nu*(2*np.random.rand(2) - 1)
        v3 *= 1 + nu*(2*np.random.rand(2) - 1)
    
        q = np.zeros([n, 6])
        p = np.zeros([n, 6])
    
        q[:, :2] = q1
        q[:, 2:4] = q2
        q[:, 4:] = q3
        p[:, :2] = v1
        p[:, 2:4] = v2
        p[:, 4:] = v3
  
        return np.hstack([p, q])
    
    def __init_data(self):
        x0 = self.random_config(self.train_traj_num + self.test_traj_num)
        X_train, y_train = self.__generate_flow(x0[:self.train_traj_num], self.h, self.train_num)
        X_test, y_test = self.__generate_flow(x0[self.train_traj_num:], self.h, self.test_num)
        self.X_train = X_train.reshape([self.train_num*self.train_traj_num, -1])
        self.y_train = y_train.reshape([self.train_num*self.train_traj_num, -1])
        self.X_test = X_test.reshape([self.test_num*self.test_traj_num, -1])
        self.y_test = y_test.reshape([self.test_num*self.test_traj_num, -1])
        
def plot(data, net):
    h_true = data.h / 10
    test_num_true = (data.test_num - 1) * 10
    if isinstance(net, ln.nn.HNN):
        flow_true = data.solver.flow(data.X_test_np[0][:-1], h_true, test_num_true)
        flow_pred = net.predict(data.X_test[0][:-1], data.h, data.test_num-1, keepinitx=True, returnnp=True)
    else:
        flow_true = data.solver.flow(data.X_test_np[0], h_true, test_num_true)
        flow_pred = net.predict(data.X_test[0], data.test_num-1, keepinitx=True, returnnp=True)
    plt.plot(flow_true[:,6], flow_true[:, 7], color='b', label='Ground truth')
    plt.plot(flow_true[:,8], flow_true[:, 9], color='b')
    plt.plot(flow_true[:,10], flow_true[:, 11], color='b')
    plt.scatter(flow_pred[:,6], flow_pred[:, 7], color='r', label='Predicted solution')
    plt.scatter(flow_pred[:,8], flow_pred[:, 9], color='r')
    plt.scatter(flow_pred[:,10], flow_pred[:, 11], color='r')
    plt.legend(loc='upper left')
    plt.savefig('three_body.pdf')

def main():
    device = 'cpu' # 'cpu' or 'gpu'
    # data
    h = 0.5
    train_num = 10
    test_num = 10
    train_traj_num = 4000
    test_traj_num = 1000
    # net
    net_type = 'LA' # 'LA' or 'G' or 'HNN'
    LAlayers = 20
    LAsublayers = 4
    Glayers = 20
    Gwidth = 50
    activation = 'sigmoid'
    Hlayers = 6
    Hwidth = 50
    Hactivation = 'tanh'
    # training
    lr = 0.001
    iterations = 300000
    print_every = 1000
    
    add_h = True if net_type == 'HNN' else False
    criterion = None if net_type == 'HNN' else 'MSE'
    data = TBData(h, train_traj_num, test_traj_num, train_num, test_num, add_h)
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