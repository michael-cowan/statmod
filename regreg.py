import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import ols

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

def test(alpha=0.5, lam=[0.1, 0.001], showfigs=True):
    """
        Tests regularized regression fitting of points from
        y = x^2 + 2x + 3
    """
    x = np.linspace(1, 5)
    y = np.poly1d([1, 2, 3])(x)# + np.random.random(len(x))
    
    bguess = np.array([4, 4, 4])

    fig, ax = plt.subplots(figsize=(11, 9))
    leg = []
    for lam in [1.0, 0.5, 0.1, 0.0001]:
        sol = elastic_net(polyfunc, x, y, bguess, alpha, lam)
        ax.plot(x, polyfunc(x, sol.x), alpha=0.75)
        leg.append('Lambda = %s' % lam)

    # solve with OLS
    sol2 = ols.ols_sing(x, y, show=False)
    ax.plot(x, polyfunc(x, sol2[1]))
    leg.append('OLS Solution')

    # plot the actual function
    ax.plot(x, y, '--', color='black')
    leg.append('Actual')
    ax.legend(leg)
    title = 'Ridge Regression' if alpha == 0 else 'LASSO' if alpha == 1 else 'Elastic Net'
    fig.canvas.set_window_title(title)
    ax.set_title('Alpha = %.2f' % alpha)
    if showfigs:
        fig.show()
    return fig, ax

if __name__ == '__main__':
    (f1, a1), (f2, a2), (f3, a3) = [test(n) for n in [0, 0.5, 1]]
