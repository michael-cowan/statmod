import numpy as np

class NeuralNetwork:
    def __init__(self, inputs, outputs, neurons=3):
        # input matrix
        self.x = np.array(inputs)
        
        # target values
        self.out = np.array(outputs)
        
        # number of input neurons
        self.nin = self.x.shape[0]
        
        # number of output neurons
        self.nout = self.out.shape[0]
        
        # weights from inputs to hidden neuron layer
        self.w1 = np.random.random([self.nin, neurons])
        
        # bias term between inputs and hidden neuron layer
        self.b1 = np.random.random()
        
        # weights from hidden layer to outputs
        self.w2 = np.random.random([neurons, self.nout])
        
        # bias term between hidden layer and outputs
        self.b2 = np.random.random()
        
        # accounts for step change within gradient descent optimization approach
        self.gamma = 0.5
        
        # network's outputs
        self.sol = self.forward()
        
        # change in weights per step
        self.change = 10.

    def sigmoidal(self, mat):
        # activation function, [0, 1] interval
        return 1 / (1 + np.exp(-mat))

    def hout(self):
        # hidden layer neurons
        return self.sigmoidal(np.dot(self.x, self.w1) + self.b1)

    def forward(self):
        # solve forwards for outputs
        return self.sigmoidal(np.dot(self.hout(), self.w2) + self.b2)

    def func(self, x=None):
        if x:
            h = self.sigmoidal(np.dot(x, self.w1) + self.b2)
            return self.sigmoidal(np.dot(h, self.w2) + self.b2)
        else:
            return self.forward()

    def backprop(self):
        # calc new w2
        self.dedo2 = self.sol - self.out
        self.dodnet2 = self.sol * (1 - self.sol)
        self.dnetdw2 = np.vstack(self.hout())
        
        # dE / dW2 solved using chain rule
        self.dedw2 = self.dedo2 * self.dodnet2 * self.dnetdw2

        # new w2 values
        self.new_w2 = self.w2 - self.gamma * self.dedw2

        #TODO: calc new w1 matrix
        # dE/dOH = dE/dO * dO/dNET * newW2
        self.dedoh = (np.dot(np.dot(self.dedo2, self.dodnet2), self.new_w2)).sum(1)
        
        # dOH/dNET = OH (1 - OH)
        self.dohdnet = self.hout() * (1 - self.hout())
        
        # dNETH1/dW = w1
        self.dneth1dw = self.w1
        
        self.dedw1 = np.dot(np.dot(self.dedoh, self.dohdnet), self.dneth1dw)
        
        self.new_w1 = self.w1 - self.gamma * self.dedw1
        
        self.sol = self.forward()

        self.change = abs((self.w1 - self.new_w1)).sum() + abs((self.w2 - self.new_w2)).sum()

        # update weights
        self.w1 = self.new_w1
        self.w2 = self.new_w2

    def solve(self, e=1E-6):
        while self.change > e:
            self.backprop()
        print 'Solution found'
        

if __name__ == '__main__':
    x = np.array([2.0, 0.4])
    y = np.array([0.3, 0.9])
    n = NeuralNetwork(x, y)
    n.solve()
    print '\nModel:'
    print n.sol
    print '\nExpected:'
    print y