import numpy as np


class NeuralNetwork:
    def __init__(self, inputs, outputs, neurons=2):
        # rinput matrix
        self.x = np.array(inputs)
        if len(self.x.shape) == 1:
            self.x = self.x.reshape(1, self.x.size)

        # target values
        self.yact = np.array(outputs)
        if len(self.yact.shape) == 1:
            self.yact = self.yact.reshape(1, self.yact.size)

        # number of inputs
        self.nin = self.x.shape[1]

        # number of outputs
        self.nout = self.yact.shape[1]

        # number of hidden neurons
        self.neurons = neurons

        # weights from inputs to hidden neuron layer
        self.w1 = np.random.random([self.nin, neurons])
        # self.w1 = np.array([[0.15, 0.25], [0.2, 0.3]])

        # bias term between inputs and hidden neuron layer
        self.b1 = np.random.random()
        # self.b1 = 0.35

        # weights from hidden layer to outputs
        self.w2 = np.random.random([neurons, self.nout])
        # self.w2 = np.array([[0.40, 0.50], [0.45, 0.55]])

        # bias term between hidden layer and outputs
        self.b2 = np.random.random()
        # self.b2 = 0.60

        # accounts for step change within gradient descent
        # optimization approach
        self.gamma = 0.5

        # calc network's outputs
        # self.forward()

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

<<<<<<< HEAD
        self.dedw2 = sum([i * np.vstack(j) for i, j in zip(self.dednet,
                                                           self.dnetdw2)])
=======
        self.dedw2 = sum([i * np.vstack(j) for i,j in zip(self.dednet, self.dnetdw2)])
>>>>>>> 8e50bdfe234090bb8fd7a7a82eef8d635d61f606

        # solve for dedw1
        self.dedw1 = np.zeros(self.w1.shape)
        self.dnetdhout = self.w2
        self.dnetdw1 = self.x
        self.dhoutdh = self.hout * (1 - self.hout)
        for i in xrange(len(self.x)):
            self.dedh = self.dednet[i, :] * self.dnetdhout

            self.dedhout = n.dedh.sum(1)
            self.dedw1 += np.matmul((self.dedhout * self.dhoutdh[i, :]).reshape(1, self.neurons), self.dnetdw1[i, :])
            # self.dedw1 = sum([np.matmul(np.vstack(self.dedhout * i), j) for i,j in zip(self.dhoutdh, self.dnetdw1)])

        self.delta_w1 = - self.gamma * self.dedw1
        self.delta_w2 = - self.gamma * self.dedw2

        self.change = abs(self.delta_w1).sum() + abs(self.delta_w2).sum()

        # update weights
        self.w1 += self.delta_w1
        self.w2 += self.delta_w2

        # calc new outputs
        self.forward()

<<<<<<< HEAD
    def train(self, e=1E-6, max_iter=10000):
        i = 0
        for i in xrange(max_iter):
            self.backprop()
            if self.change > e:
                print('Training converged')
                break
        else:
            print('Did not converge.')


if __name__ == '__main__':
    x = np.array([[0.05, 0.10]])
    # x = np.vstack(range(1, 5))
    x.sort()
    # yact = x ** 2
    yact = np.array([[0.01, 0.99]])
    ymax = float(yact.max())
    y = yact   # / ymax
    n = NeuralNetwork(x, y)
    n.train()
    if 1:
        print '\nModel:'
        print n.y
        print '\nExpected:'
        print n.yact
=======
    def solve(self, e=1E-6):
        while self.change > e:
            self.backprop()
        print('Solution found')

if __name__ == '__main__':
    x = np.array([0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35])
    x = np.vstack(x)
    yact = 4 - 2 * np.cos(x)
    ymax = float(yact.max())
    y = yact
    n = NeuralNetwork(x, y)
    # n.solve()
    if 0:
        print('\nModel:')
        print(n.y)
        print('\nExpected:')
        print(n.yact)
>>>>>>> 8e50bdfe234090bb8fd7a7a82eef8d635d61f606
        import matplotlib.pyplot as plt
        plt.plot(n.yact, n.y, '.')
        plt.plot(np.linspace(0, 1), np.linspace(0, 1))
        plt.xlabel('Actual')
        plt.ylabel('Model')
        plt.show()
