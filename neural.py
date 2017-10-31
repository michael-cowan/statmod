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

    def sigmoidal(self, mat):
        # activation function, [0, 1] interval
        return 1 / (1 + np.exp(-mat))

    def hout(self):
        # hidden layer neurons
        return self.sigmoidal(np.dot(self.x, self.w1) + self.b1)

    def forward(self):
        # solve forwards for outputs
        return self.sigmoidal(np.dot(self.hout(), self.w2) + self.b2)

    def backprop(self):
        # calc new w2
        dedo2 = self.sol - self.out
        dodnet2 = self.sol * (1 - self.sol)
        dnetdw2 = np.vstack(self.hout())
        
        # dE / dW2 solved using chain rule
        dedw2 = dedo2 * dodnet2 * dnetdw2

        # new w2 values
        self.new_w2 = self.w2 - self.gamma * dedw2

        #TODO: calc new w1 matrix

if __name__ == '__main__':
    x = np.array([2.0, 0.4])
    y = np.array([0.5, 3.2])
    n = NeuralNetwork(x, y)