"""
@author: Pengzhan Jin (jpz@pku.edu.cn)
"""
import torch
import numpy as np
from itertools import product
from .module import Map
from .fnn import FNN

class MIONet(Map):
    '''Multiple-input operator network.
    Input: ([batch, sensors1], [batch, sensors2],..., [batch, dim_loc])
    Output: [batch, 1]
    '''
    def __init__(self, sizes, activation='relu', initializer='default', bias=True):
        super(MIONet, self).__init__()
        self.sizes = sizes
        self.activation = activation
        self.initializer = initializer
        self.bias = bias

        self.ms = self.__init_modules()
        self.ps = self.__init_parameters()
    
    def forward(self, x):
        y = torch.stack([self.ms['Net{}'.format(i + 1)](x[i]) for i in range(len(self.sizes))])
        y = torch.sum(torch.prod(y, dim=0), dim=-1, keepdim=True)
        return y + self.ps['bias'] if self.bias else y
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        for i in range(len(self.sizes)):
            modules['Net{}'.format(i + 1)] = FNN(self.sizes[i], self.activation, self.initializer)
        return modules
    
    def __init_parameters(self):
        parameters = torch.nn.ParameterDict()
        if self.bias:
            parameters['bias'] = torch.nn.Parameter(torch.zeros([1]))
        return parameters
    
class MIONet_Cartesian(Map):
    '''Multiple-input operator network (Cartesian product version).
    Input: ([batch, sensors1], [batch, sensors2],..., [(batch,) num_loc, dim_loc])
    Output: [batch, num_loc]
    '''
    def __init__(self, sizes, activation='relu', initializer='default', bias=True):
        super(MIONet_Cartesian, self).__init__()
        self.sizes = sizes
        self.activation = activation
        self.initializer = initializer
        self.bias = bias

        self.ms = self.__init_modules()
        self.ps = self.__init_parameters()
    
    def forward(self, x):
        y1 = torch.stack([self.ms['Net{}'.format(i + 1)](x[i]) for i in range(len(self.sizes) - 1)])
        y1 = torch.prod(y1, dim=0)
        y2 = self.ms['Net{}'.format(len(self.sizes))](x[-1])
        if len(y2.size()) == 3:
            y = torch.einsum('ij,ikj->ik', y1, y2)
        else:
            y = y1 @ y2.t()
        return y + self.ps['bias'] if self.bias else y
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        for i in range(len(self.sizes)):
            modules['Net{}'.format(i + 1)] = FNN(self.sizes[i], self.activation, self.initializer)
        return modules
    
    def __init_parameters(self):
        parameters = torch.nn.ParameterDict()
        if self.bias:
            parameters['bias'] = torch.nn.Parameter(torch.zeros([1]))
        return parameters

class MIONet_precomp(Map):
    '''Multiple-input operator network (pre-computing version).
    Input: ([batch, sensors1], [batch, sensors2],..., [(batch,) num_loc, dim_loc])
    Output: [batch, num_loc]
    '''
    def __init__(self, sizes, intervals, dpis, activation='relu', initializer='default', bias=True):
        super(MIONet_precomp, self).__init__()
        self.sizes = sizes
        self.intervals = intervals
        self.dpis = dpis
        self.activation = activation
        self.initializer = initializer
        self.bias = bias

        self.ms = self.__init_modules()
        self.ps = self.__init_parameters()
        self.pre_comp_points_cache = None
    
    def forward(self, x):
        y1 = torch.stack([self.ms['Net{}'.format(i + 1)](x[i]) for i in range(len(self.sizes) - 1)])
        y1 = torch.prod(y1, dim=0)
        pts = x[-1]
        if len(pts.size()) == 2:
            y2 = self.ms['Net{}'.format(len(self.sizes))](pts)
            y = y1 @ y2.t()
        elif (pts.size()[:-1]).numel() < (torch.Size(self.dpis)).numel():
            #### no pre-computing
            y2 = self.ms['Net{}'.format(len(self.sizes))](pts)
            y = torch.einsum('ij,ikj->ik', y1, y2)
        else:
            #### pre-computing
            y2 = self.ms['Net{}'.format(len(self.sizes))](self.pre_comp_points)
            y = (y1 @ y2.t()).view(y1.size(0), *self.dpis)
            mask = pts - torch.tensor([interval[0] for interval in self.intervals], 
                                      device=pts.device, dtype=pts.dtype)
            mask = mask / torch.tensor([(self.intervals[i][1] - self.intervals[i][0]) / (self.dpis[i] - 1)
                                        for i in range(len(self.intervals))], 
                                       device=pts.device, dtype=pts.dtype)
            mask = torch.round(mask).int()
            y = y[torch.arange(y1.size(0)).unsqueeze(1), *[mask[..., i] for i in range(mask.size(-1))]]
        return y + self.ps['bias'] if self.bias else y
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        for i in range(len(self.sizes)):
            modules['Net{}'.format(i + 1)] = FNN(self.sizes[i], self.activation, self.initializer)
        return modules
    
    def __init_parameters(self):
        parameters = torch.nn.ParameterDict()
        if self.bias:
            parameters['bias'] = torch.nn.Parameter(torch.zeros([1]))
        return parameters
    
    @property
    def pre_comp_points(self):
        if (isinstance(self.pre_comp_points_cache, torch.Tensor) and 
            (self.pre_comp_points_cache.device == self.device) and
            (self.pre_comp_points_cache.dtype == self.dtype)):
            pass
        else:
            dim = len(self.intervals)
            itvs = []
            for i in range(dim):
                itvs.append(np.linspace(self.intervals[i][0], self.intervals[i][1], num=self.dpis[i]))
            x = np.array(list(product(*itvs)))
            self.pre_comp_points_cache = torch.tensor(x, dtype=self.dtype, device=self.device)
        return self.pre_comp_points_cache