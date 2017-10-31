import numpy as np

class NeuralNetwork:
    def __init__(self, inputs, outputs, neurons=3):
        self.x = np.array(inputs)
        self.out = np.array(outputs)
        
        self.nin = self.x.shape[0]
        self.nout = self.out.shape[0]
        
        self.w1 = np.random.random([self.nin, neurons])
        self.w2 = np.random.random([neurons, self.nout])
        self.b1 = np.random.random()
        self.b2 = np.random.random()
        
        self.gamma = 0.5
        
        self.sol = self.forward()

    def sigmoidal(self, mat):
        return 1 / (1 + np.exp(-mat))

    def hout(self):
        return self.sigmoidal(np.dot(self.x, self.w1) + self.b1)

    def forward(self):
        return self.sigmoidal(np.dot(self.hout(), self.w2) + self.b2)

    def backprop(self):
        # calc new w2
        dedo2 = self.sol - self.out
        dodnet2 = self.sol * (1 - self.sol)
        dnetdw2 = np.vstack(self.hout())
        
        dedw2 = dedo2 * dodnet2 * dnetdw2

        self.new_w2 = self.w2 - self.gamma * dedw2



a = NeuralNetwork(x, y)