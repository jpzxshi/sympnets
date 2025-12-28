"""
@author: Pengzhan Jin (jpz@pku.edu.cn)
"""
import numpy as np
import matplotlib.pyplot as plt
import learner as ln
from learner.integrator.hamiltonian import SV
    
class PDData_irreg(ln.Data):
    '''Data of irregular time step for pendulum system with the Hamiltonian H(p,q)=(1/2)p^2−cos(q).
    '''
    def __init__(self, p_interval, q_interval, h_train_interval, x0_test, h_test, train_num, test_num):
        super(PDData_irreg, self).__init__()
        self.p_interval = p_interval
        self.q_interval = q_interval
        self.h_train_interval = h_train_interval
        self.x0_test = x0_test
        self.h_test = h_test
        self.train_num = train_num
        self.test_num = test_num
        
        self.dH = lambda p, q: (p, np.sin(q))
        self.solver = SV(None, self.dH, iterations=1, order=6, N=50)
        
        self.__init_data()
        
    @property
    def dim(self):
        return 2

    def __generate_flow(self, x0, h, num, add_h=False):
        variant_h = isinstance(h, list) or isinstance(h, tuple) or isinstance(h, np.ndarray)
        h_min, h_max = h if variant_h else (h, h)
        H = np.random.rand(num) * (h_max - h_min) + h_min
        X = [np.array(x0)]
        for i in range(num):
            X.append(self.solver.solve(X[-1], H[i]))
        X = np.array(X)
        return ([X[:-1, :], H[:, None]], X[1:, :]) if variant_h or add_h else (X[:-1, :], X[1:, :])
    
    def __generate_irreg(self, p_interval, q_interval, h_interval, num):
        P = np.random.rand(num) * (p_interval[1] - p_interval[0]) + p_interval[0]
        Q = np.random.rand(num) * (q_interval[1] - q_interval[0]) + q_interval[0]
        H = np.random.rand(num) * (h_interval[1] - h_interval[0]) + h_interval[0]
        X = np.hstack([P[:, None], Q[:, None]])
        y = np.array(list(map(lambda x, h: self.solver.solve(x, h), X, H)))
        return [X, H[:, None]], y
        
    def __init_data(self):
        self.X_train, self.y_train = self.__generate_irreg(self.p_interval, self.q_interval, self.h_train_interval, self.train_num)
        self.X_test, self.y_test = self.__generate_flow(self.x0_test, [self.h_test, self.h_test], self.test_num)

def plot(data, net):
    steps = 1000
    flow_true = data.solver.flow(data.X_test_np[0][0], data.h_test, steps)
    if isinstance(net, ln.nn.HNN):
        flow_pred = net.predict(data.X_test[0][0], data.h_test, steps, keepinitx=True, returnnp=True)
    else:
        flow_pred = net.predict([data.X_test[0][0], data.h_test], steps, keepinitx=True, returnnp=True)
    
    plt.plot(flow_true[:, 0], flow_true[:, 1], color='b', label='Ground truth', zorder=0)
    plt.plot(flow_pred[:, 0], flow_pred[:, 1], color='r', label='Predicted flow', zorder=1)
    plt.legend()
    plt.savefig('irregular.pdf')

def main():
    device = 'cpu' # 'cpu' or 'gpu'
    # data
    p_interval = [-np.sqrt(2), np.sqrt(2)]
    q_interval = [-np.pi / 2, np.pi / 2]
    x0_test = [0, 1]
    h_train_interval = [0.2, 0.5]
    h_test = 0.1
    train_num = 40
    test_num = 100
    # SympNet
    net_type = 'LA' # 'LA' or 'G' or 'HNN'
    LAlayers = 5
    LAsublayers = 5
    Glayers = 5
    Gwidth = 30
    activation = 'sigmoid'
    Hlayers = 4
    Hwidth = 30
    Hactivation = 'tanh'
    Hinitializer = 'orthogonal'
    # training
    lr = 0.01
    iterations = 50000
    print_every = 1000
    
    data = PDData_irreg(p_interval, q_interval, h_train_interval, x0_test, h_test, train_num, test_num)
    if net_type == 'LA':
        net = ln.nn.LASympNet(data.dim, LAlayers, LAsublayers, activation)
    elif net_type == 'G':
        net = ln.nn.GSympNet(data.dim, Glayers, Gwidth, activation)
    elif net_type == 'HNN':
        net = ln.nn.HNN([data.dim] + [Hwidth] * (Hlayers - 1) + [1], Hactivation, Hinitializer, integrator='midpoint')
    criterion = None if net_type == 'HNN' else 'MSE'

    kwargs = {
        'data': data,
        'net': net,
        'criterion': criterion,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': None,
        'print_every': print_every,
        'save': 'best_only',
        'callback': None,
        'dtype': 'float',
        'device': device
    }
    
    ln.Brain.Init(**kwargs)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output() # ln.Brain.Output(data=False)
    
    plot(data, ln.Brain.Best_model())


if __name__ == '__main__':
    main()