import numpy as np
import matplotlib.pyplot as plt

"""
Example Artifical Neural Network
- constrained to 1 input, 1 hidden layer, and 1 output
- can adjust number of nodes in hidden layer
- uses batch backpropagation to train
"""


class NeuralNetwork:
    def __init__(self, inputs, outputs, neurons=2, activation='sigmoid'):
        # input matrix
        self.x = np.vstack(inputs)

        # target values
        self.yact = np.vstack(outputs)

        # predicted values
        self.y = None

        # define activation function and differential func
        self.activation = getattr(self, activation)
        self.diff_activation = getattr(self, 'diff_' + activation)

        # number of inputs
        self.nin = self.x.shape[1]

        # number of outputs
        self.nout = self.yact.shape[1]

        # number of hidden neurons
        self.neurons = neurons

        # weights from inputs to hidden neuron layer
        self.w1 = 2 * np.random.random([self.nin, neurons]) - 1

        # bias term between inputs and hidden neuron layer
        self.b1 = np.random.random()

        # weights from hidden layer to outputs
        self.w2 = 2 * np.random.random([neurons, self.nout]) - 1

        # bias term between hidden layer and outputs
        self.b2 = np.random.random()

        # calc network's outputs
        self.forward()

    def __call__(self, x):
        h = self.activation(np.dot(np.vstack(x), self.w1) + self.b2)
        return self.activation(np.dot(h, self.w2) + self.b2)

    def relu(self, arr):
        return arr * (arr > 0).astype(float)

    def diff_relu(self, arr):
        return (arr > 0).astype(float)

    def sigmoid(self, arr):
        # activation function, [0, 1] interval
        return 1 / (1 + np.exp(-arr))

    def diff_sigmoid(self, arr):
        # derivative of activation function
        return np.exp(-arr) / (1 + np.exp(-arr))**2

    def forward(self):
        # solve forwards for outputs
        self.h = np.dot(self.x, self.w1) + self.b1
        self.hout = self.activation(self.h)
        self.net = np.dot(self.hout, self.w2) + self.b2
        self.y = self.activation(self.net)
        self.e = (0.5 * (self.y - self.yact)**2).mean()

    def backprop(self, learning_rate):
        # solve for (de / dw2) = (de / dy) (dy / dnet) (dnet / dw2)
        self.dedy = (self.y - self.yact)
        self.dydnet = self.diff_activation(self.net)  # self.y * (1 - self.y)        
        self.dnetdw2 = self.hout
        self.dedw2 = np.matmul(self.dedy * self.dydnet.T, self.dnetdw2)

        # solve for (de/dw1) = (de/dy) (dy / dw1)
        self.dhdw1 = self.x
        self.dnetdw1 = self.diff_activation(self.h) * self.dhdw1 * self.w2.T
        self.dydw1 = self.diff_activation(self.dnetdw1)
        self.dedw1 = self.dedy * self.dydw1

        # average the (de / dw) values across the batch
        self.delta_w1 = self.dedw1.mean(0, keepdims=True)
        self.delta_w2 = self.dedw2.T.mean(1, keepdims=True)

        # update weights
        self.w1 -= learning_rate * self.delta_w1
        self.w2 -= learning_rate * self.delta_w2

    def train(self, epochs=100, batch_size=16, learning_rate=0.01):
        X = self.x.copy()
        yact = self.yact.copy()
        res = np.zeros((epochs, 2))
        res[:, 0] = range(epochs)
        for epoch in range(epochs):
            for j in range(0, X.shape[0], batch_size):
                self.x = X[j: j + batch_size, :]
                self.yact = yact[j: j + batch_size, :]
                self.forward()
                self.backprop(learning_rate)
            res[epoch, 1] = self.e
            print(f'Epoch {epoch + 1:05d}: MSE = {self.e:.5f}')
        self.x = X
        self.yact = yact
        self.forward()
        return res


if __name__ == '__main__':
    np.random.seed(15224)

    x = np.random.random(10_000) * 8

    # fit NN to a polynomial
    y = x ** 2 + 3 * x

    # normalize outputs: [0, 1]
    # since we're using a sigmoid activation on the output layer
    y /= y.max()

    # initialize and train neural network
    nn = NeuralNetwork(x, y, 128)
    results = nn.train()

    # plot a learning curve
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(results[:, 0], results[:, 1])
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Learning Curve')

    # make parity plot
    plt.subplot(122)
    plt.plot(nn.yact, nn.y, '.')
    plt.plot([0, 1], [0, 1], color='k', zorder=-100)
    plt.xlabel('Actual')
    plt.ylabel('Model')
    plt.title('Parity Plot')
    plt.legend([], title=f'MSE = {nn.e:.3e}', frameon=False)
    plt.tight_layout()
    plt.show()
