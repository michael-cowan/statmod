import numpy as np

class NeuralNetwork:
    def __init__(self, inputs, outputs, neurons=3):
        # input matrix
        self.x = np.array(inputs)
        if len(self.x.shape) == 1:
            self.x = self.x.reshape(1, self.x.size)
        
        # target values
        self.yact = np.array(outputs)
        if len(self.yact.shape) == 1:
            self.yact = self.yact.reshape(1, self.yact.size)
        
        # number of input neurons
        self.nin = self.x.shape[1]
        
        # number of output neurons
        self.nout = self.yact.shape[1]
        
        # weights from inputs to hidden neuron layer
        self.w1 = np.random.random([self.nin, neurons])
        #self.w1 = np.array([[0.15, 0.25], [0.2, 0.3]])

        # bias term between inputs and hidden neuron layer
        self.b1 = np.random.random()
        #self.b1 = 0.35
        
        # weights from hidden layer to outputs
        self.w2 = np.random.random([neurons, self.nout])
        #self.w2 = np.array([[0.40, 0.50], [0.45, 0.55]])
        
        # bias term between hidden layer and outputs
        self.b2 = np.random.random()
        #self.b2 = 0.60
        
        # accounts for step change within gradient descent optimization approach
        self.gamma = 0.5
        
        # calc network's outputs
        self.forward()
        
        # change in weights per step
        self.change = 10.

    def sigmoidal(self, mat):
        # activation function, [0, 1] interval
        return 1 / (1 + np.exp(-mat))

    def diff_sigmoidal(self, mat):
        # derivative of activation function
        return np.exp(-mat) / (1 + np.exp(-mat))**2

    def forward(self):
        # solve forwards for outputs
        self.h = np.dot(self.x, self.w1) + self.b1
        self.hout = self.sigmoidal(self.h)
        self.net = np.dot(self.hout, self.w2) + self.b2
        self.y = self.sigmoidal(self.net)

    def func(self, x=None):
        if x:
            h = self.sigmoidal(np.dot(x, self.w1) + self.b2)
            return self.sigmoidal(np.dot(h, self.w2) + self.b2)
        else:
            self.forward()
            return self.y

    def backprop(self):
        # solve for dedw2
        self.dedy = self.y - self.yact
            
        self.dydnet = self.y * (1 - self.y)
        
        self.dednet = (self.dedy * self.dydnet)
            
        self.dnetdw2 = self.hout
        
        self.dedw2 = sum([i * np.vstack(j) for i,j in zip(self.dednet, self.dnetdw2)])
        
        # solve for dedw1
        self.dedw1 = np.zeros(self.w1.shape)
        self.dnetdhout = self.w2
        self.dnetdw1 = self.x
        self.dhoutdh = self.hout * (1 - self.hout)
        for i in xrange(len(self.x)):
            self.dedh = self.dednet[i, :] * self.dnetdhout
            
            self.dedhout = n.dedh.sum(1)
            self.dedw1 += np.matmul(np.vstack(self.dedhout * self.dhoutdh[i, :]), self.dnetdw1[i, :]) 
            #self.dedw1 = sum([np.matmul(np.vstack(self.dedhout * i), j) for i,j in zip(self.dhoutdh, self.dnetdw1)])

        self.delta_w1 = - self.gamma * self.dedw1
        self.delta_w2 = - self.gamma * self.dedw2

        self.change = abs(self.delta_w1).sum() + abs(self.delta_w2).sum()

        # update weights
        self.w1 += self.delta_w1
        self.w2 += self.delta_w2

        # calc new outputs
        self.forward()
        
    def solve(self, e=1E-5):
        while self.change > e:
            self.backprop()
        print 'Solution found'

if __name__ == '__main__':
    #x = np.array([[0.05, 0.10]])
    x = np.vstack(range(1, 5))
    x.sort()
    yact = 0.08 * x
    #yact = np.array([[0.01, 0.99]])
    #ymax = float(yact.max())
    y = yact# / ymax
    n = NeuralNetwork(x, y)
    n.solve()
    if 1:
        print '\nModel:'
        print n.y
        print '\nExpected:'
        print n.yact
        import matplotlib.pyplot as plt
        plt.plot(n.yact, n.y, '.')
        plt.xlabel('Actual')
        plt.ylabel('Model')
        plt.show()