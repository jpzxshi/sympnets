"""
@author: jpzxshi
"""
import numpy as np
from .data import Data
from ..utils import map_elementwise

class Data_MIONet_Cartesian(Data):
    '''Data format for MIONet (Cartesian product version).
    '''
    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None):
        super(Data_MIONet_Cartesian, self).__init__(X_train, y_train, X_test, y_test)
        
    def get_batch(self, batch_size):
        @map_elementwise
        def batch_mask(X, num):
            return np.random.choice(X.size(0), num, replace=False)
        @map_elementwise
        def batch(X, mask):
            return X[mask]
        mask = batch_mask(self.y_train, batch_size)
        if len(self.X_train[-1].size()) == 3:
            return batch(self.X_train, mask), batch(self.y_train, mask)
        else:
            return (*batch(self.X_train[:-1], mask), self.X_train[-1]), batch(self.y_train, mask)