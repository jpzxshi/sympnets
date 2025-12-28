"""
@author: jpzxshi
"""
import os
import numpy as np
import torch
from .nn import Algorithm
from .utils import timing, str_current_time, cross_entropy_loss

class Brain:
    '''Runner based on torch.
    '''
    brain = None
    
    @classmethod
    def Init(cls, data, net, criterion='MSE', optimizer='Adam', lr=0.01, 
             iterations=100, batch_size=None, print_every=10, save='best_only', 
             callback=None, dtype='float', device='cpu'):
        cls.brain = cls(data, net, criterion, optimizer, lr, iterations, 
                        batch_size, print_every, save, callback, dtype, device)
    
    @classmethod
    def Run(cls, **kwargs):
        cls.brain.run(**kwargs)
    
    @classmethod
    def Restore(cls):
        cls.brain.restore()
    
    @classmethod
    def Output(cls, data=True, best_model=True, loss_history=True, info=None, path=None, **kwargs):
        cls.brain.output(data, best_model, loss_history, info, path, **kwargs)
    
    @classmethod
    def Loss_history(cls):
        return cls.brain.loss_history_list
    
    @classmethod
    def Best_model(cls):
        return cls.brain.best_model
    
    @classmethod
    def Clear(cls):
        cls.brain = None
    
    @classmethod
    def Start(cls, msg='Start'):
        cls.Clear()
        print(msg + ' (' + str_current_time() + ')\nInitializing...', flush=True)
    
    @classmethod
    def End(cls, msg='End'):
        cls.Clear()
        print(msg + ' (' + str_current_time() + ')', flush=True)
    
    def __init__(self, data, net, criterion='MSE', optimizer='Adam', lr=0.01, 
                 iterations=100, batch_size=None, print_every=10, save='best_only', 
                 callback=None, dtype='float', device='cpu'):
        self.data = data
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.iterations = iterations
        self.batch_size = batch_size
        self.print_every = print_every
        self.save = save
        self.callback = callback
        self.dtype = dtype
        self.device = device
        
        self.loss_history_list = []
        self.loss_history = None
        self.best_model = None
        
        self.__optimizer = None
        self.__criterion = None
        
        print('Initialized: \'data\' and \'net\' are ready (' + str_current_time() + ')', flush=True)
    
    @timing
    def run(self, **kwargs):
        for key, arg in kwargs.items():
            setattr(self, key, arg)
        self.__init_brain()
        if len(self.loss_history_list) == 0:
            print('Training... (' + str_current_time() + ')', flush=True)
        else:
            print('Training... (Round {}) ('.format(len(self.loss_history_list) + 1) + str_current_time() + ')', flush=True)
        loss_history = []
        best_model_index = 0
        for i in range(self.iterations + 1):
            if self.batch_size is None:
                X_train, y_train = self.data.X_train, self.data.y_train
            else:
                X_train, y_train = self.data.get_batch(self.batch_size)
            if i % self.print_every == 0 or i == self.iterations:
                if isinstance(self.net, Algorithm):
                    loss_train = self.__criterion(self.net(X_train), y_train)
                    loss_test = self.__criterion(self.net(self.data.X_test), self.data.y_test)
                else:
                    with torch.inference_mode():
                        loss_train = self.__criterion(self.net(X_train), y_train)
                        loss_test = self.__criterion(self.net(self.data.X_test), self.data.y_test)
                loss_history.append([i, loss_train.item(), loss_test.item()])
                print('{:<9}Train loss: {:<25}Test loss: {:<25}'.format(i, loss_train.item(), loss_test.item()), flush=True)
                if torch.any(torch.isnan(loss_train)):
                    raise RuntimeError('encountering nan, stop training')
                if self.save == False:
                    pass
                elif self.save in [True, 'best_only', 'best_only_test', 'best_only_train', 'all']:
                    if not os.path.exists('model'): os.mkdir('model')
                    if self.save == 'all':
                        torch.save(self.net, 'model/model{}.pkl'.format(i))
                    else:
                        index_temp = 1 if self.save == 'best_only_train' else 2
                        if loss_history[-1][index_temp] < loss_history[best_model_index][index_temp] or len(loss_history) == 1:
                            best_model_index = len(loss_history) - 1
                            torch.save(self.net, 'model/model_best.pkl')
                else:
                    raise ValueError
                if self.callback is not None:
                    to_stop = self.callback(self.data, self.net)
                    if to_stop: break
            if i < self.iterations:
                if self.optimizer in ['LBFGS']:
                    def closure():
                        self.__optimizer.zero_grad()
                        loss = self.__criterion(self.net(X_train), y_train)
                        loss.backward()
                        return loss
                    self.__optimizer.step(closure)
                else:
                    self.__optimizer.zero_grad()
                    loss = self.__criterion(self.net(X_train), y_train)
                    loss.backward()
                    self.__optimizer.step()
        self.loss_history = np.array(loss_history)
        self.loss_history_list.append(self.loss_history)
        print('Done!', flush=True)
        return self.loss_history
    
    def restore(self):
        if self.loss_history is not None and self.save != False:
            index_temp = 1 if self.save == 'best_only_train' else 2
            best_loss_index = np.argmin(self.loss_history[:, index_temp])
            iteration = int(self.loss_history[best_loss_index, 0])
            loss_train = self.loss_history[best_loss_index, 1]
            loss_test = self.loss_history[best_loss_index, 2]
            print('Best model at iteration {}:'.format(iteration), flush=True)
            print('Train loss:', loss_train, 'Test loss:', loss_test, flush=True)
            path = 'model/model{}.pkl'.format(iteration) if self.save == 'all' else 'model/model_best.pkl'
            self.best_model = torch.load(path, weights_only=False)
            self.net = self.best_model
        else:
            raise RuntimeError('restore before running or without saved model')
        return self.best_model
    
    def output(self, data, best_model, loss_history, info, path, **kwargs):
        if path is None:
            path = './outputs/' + str_current_time()
        if not os.path.isdir(path): os.makedirs(path)
        if data:
            self.data.save(path)
        if best_model:
            torch.save(self.best_model, path + '/model_best.pkl')
        if loss_history:
            if len(self.loss_history_list) == 1:
                np.savetxt(path + '/loss.txt', self.loss_history)
            else:
                for i in range(len(self.loss_history_list)):
                    np.savetxt(path + '/loss{}.txt'.format(i + 1), self.loss_history_list[i])
        if info is not None:
            with open(path + '/info.txt', 'w') as f:
                for key, arg in info.items():
                    f.write('{}: {}\n'.format(key, str(arg)))
        for key, arg in kwargs.items():
            np.savetxt(path + '/' + key + '.txt', arg)
        print('The results have been output to dir \'' + path + '\'', flush=True)
    
    def __init_brain(self):
        self.loss_history = None
        self.best_model = None
        self.data.device = self.device
        self.data.dtype = self.dtype
        self.net.device = self.device
        self.net.dtype = self.dtype
        self.__init_optimizer()
        self.__init_criterion()
    
    def __init_optimizer(self):
        if self.optimizer in ['Adam', 'adam']:
            self.__optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        elif self.optimizer == 'LBFGS':
            self.__optimizer = torch.optim.LBFGS(self.net.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
    
    def __init_criterion(self):
        if isinstance(self.net, Algorithm):
            self.__criterion = self.net.criterion
            if self.criterion is not None:
                import warnings
                warnings.warn('loss-oriented neural network has already implemented its loss function')
        elif self.criterion == 'MSE':
            self.__criterion = torch.nn.MSELoss()
        elif self.criterion == 'CrossEntropy':
            self.__criterion = cross_entropy_loss
        else:
            raise NotImplementedError