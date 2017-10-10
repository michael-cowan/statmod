import numpy as np
from scipy.optimize import minimize

def elastic_net(func, x, y, bguess, alpha, lam):

    def loss(b):
        return (1 / (2. * len(x))) * ((func(x, b) - y)**2).sum()

    def tot_func(b, alpha, lam):
        bnorm = b.sum()
        bnorm2 = (b**2).sum()
        return loss(b) + lam * ((alpha * bnorm) + ((1 - alpha) * bnorm2))

    return minimize(tot_func, bguess, (alpha, lam))

if __name__ == '__main__':
    def func(x, b): return np.poly1d(b)(x)

    x = np.linspace(0, 5, 100)
    y = np.poly1d([1, 2, 3])(x) + np.random.random(len(x))
    
    bguess = np.array([4, 4, 4])
    
    alpha = 0.5
    lam = 0.01
    
    import matplotlib.pyplot as plt
    leg = []
    for lam in [0.1, 0.001]:
        sol = elastic_net(func, x, y, bguess, alpha, lam)
        plt.plot(x, func(x, sol.x), '.')
        leg.append(str(lam))

    plt.plot(x, func(x, [1, 2, 3]))
    leg.append('Actual')
    plt.legend(leg)
    plt.show()
