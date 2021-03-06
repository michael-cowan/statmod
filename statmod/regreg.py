import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import statmod.ols as ols


def elastic_net(func, x, y, bguess, alpha, lam, const_alpha=False):
    bguess = np.array(bguess)
    # Add in alpha and lambda into variables of the loss function
    if not const_alpha:
        terms = np.array(bguess.tolist() + [alpha, lam])
    else:
        terms = bguess

    def loss(b):
        return (1 / (2. * len(x))) * ((func(x, b) - y)**2).sum()

    def tot_func(terms, alpha=alpha, lam=lam):
        if not const_alpha:
            b = terms[:-2]
            alpha = terms[-2]
            lam = terms[-1]
        else:
            b = terms
        bnorm = b.sum()
        bnorm2 = (b**2).sum()
        return (loss(b) + lam * ((alpha * bnorm) + ((1 - alpha) * bnorm2))).sum()

    # Constraints: 1 >= alpha >= 0, lambda >= 0
    cons = [{'type': 'ineq',
             'fun': lambda x: x[-2]},
            {'type': 'ineq',
             'fun': lambda x: 1 - x[-2]},
            {'type': 'ineq',
             'fun': lambda x: x[-1]}
            ]

    if const_alpha:
        cons = None

    sol = minimize(tot_func, terms, constraints=cons)
    if sol.success:
        bf = sol.x
        if not const_alpha:
            bf = bf[:-2]

        # model output
        y_mod = func(x, bf)

        # calc total sum of squares
        tot_sos = ((y - y.mean())**2).sum()

        # calc regression sum of squares
        resid_sos = ((y - y_mod)**2).sum()
        reg_sos = tot_sos - resid_sos

        # calculate correlation coefficient, R2
        r_sq = reg_sos / tot_sos

    else:
        r_sq = 0

    sol.r_sq = r_sq
    sol.resid = y - y_mod
    sol.mse = (sol.resid**2).mean()

    return sol


def polyfunc(x, b):
    """
        y = (b0)x^N + (b1)x^(N-1) + ... + bN
    """
    return np.poly1d(b)(x)


def test(alpha=0.5, lam=0.1, testfuncb=[2, 3, 4], showfigs=True):
    """
        Tests regularized regression fitting of points from
        y = 2x^2 + 3x + 4 + (error)
    """
    x = np.linspace(1, 5)
    y = polyfunc(x, testfuncb) + 5*np.random.random(len(x))
    
    bguess = np.array([1]*6)

    fig, ax = plt.subplots(figsize=(11, 9))
    leg = []
    sol = elastic_net(polyfunc, x, y, bguess, alpha, lam)
    b = sol.x[:-2]
    alpha = round(sol.x[-2], 2)
    lam = sol.x[-1]
    ax.plot(x, polyfunc(x, b))
    leg.append('Lambda = %.3e' % lam)

    # solve with OLS
    sol2 = ols.ols_sing(x, y, show=False)
    ax.plot(x, polyfunc(x, sol2.b))
    leg.append('OLS Solution')

    # plot the actual function
    ax.plot(x, y, '--', color='black')
    leg.append('Actual')
    ax.legend(leg)
    title = 'Ridge Regression' if alpha == 0 else 'LASSO' if alpha == 1 else 'Elastic Net'
    fig.canvas.set_window_title(title)
    ax.set_title('%s\nAlpha = %.2f' % (title, alpha))
    if showfigs:
        plt.show()
    return fig, ax, sol, sol2

if __name__ == '__main__':
    fig, ax, rr_sol, ols_sol = test()
