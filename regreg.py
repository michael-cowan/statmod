import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def elastic_net(func, x, y, bguess, alpha, lam):

    def loss(b):
        return (1 / (2. * len(x))) * ((func(x, b) - y)**2).sum()

    def tot_func(b, alpha, lam):
        bnorm = b.sum()
        bnorm2 = (b**2).sum()
        return loss(b) + lam * ((alpha * bnorm) + ((1 - alpha) * bnorm2))

    return minimize(tot_func, bguess, (alpha, lam))

def polyfunc(x, b):
    """
        y = (b0)x^N + (b1)x^(N-1) + ... + bN
    """
    return np.poly1d(b)(x)

def test(alpha=0.5, lam=[0.1, 0.001]):
    """
        Tests regularized regression fitting of points from
        y = x^2 + 2x + 3
    """
    x = np.linspace(0, 5, 100)
    y = np.poly1d([1, 2, 3])(x) + np.random.random(len(x))
    
    bguess = np.array([4, 4, 4])

    leg = []
    for lam in [0.1, 0.001]:
        sol = elastic_net(polyfunc, x, y, bguess, alpha, lam)
        plt.plot(x, polyfunc(x, sol.x))
        leg.append('Lambda = %s' % lam)

    plt.plot(x, polyfunc(x, [1, 2, 3]), '--', color='black')
    leg.append('Actual')
    plt.legend(leg)
    title = 'Ridge Regression' if alpha == 0 else 'LASSO' if alpha == 1 else 'Elastic Net\nAlpha = %.2f' % alpha
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    test()
