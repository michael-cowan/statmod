import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


class Solution:
    def __init__(self, format, b, pval, r2, mse, name, resid):
        self.info = {'format': 'string representation of model',
                     'b': 'model coefficients',
                     'pval': 'p distribution numbers',
                     'r2': 'correlation coefficient (R^2)',
                     'mse': 'mean square error',
                     'sum': 'pandas DataFrame summary of results',
                     'name': 'info on inputs being used',
                     'func': 'function to use model by passing in inputs',
                     'resid': 'target - output'
                     }

        self.format = format
        self.b = b
        self.pval = pval
        self.r2 = r2
        self.mse = mse
        self.name = name
        self.resid = resid
        if format:
            self.sum = pd.DataFrame({'Format': self.format.split(', '),
                                    'Coefficient': self.b.T.tolist(),
                                    'PValue': self.pval.tolist()
                                    })[['Format', 'Coefficient', 'PValue']]
        else:
            self.sum = pd.DataFrame()

    def func(self, xin):
        if self.sum.empty:
            print 'This regression resulted in an exception. A solution was not found'
            return
        if isinstance(xin, int) or isinstance(xin, float):
            xin = [xin]
        ans = 0
        for i, row in self.sum.iterrows():
            if '*' in row.Format:
                n1, n2 = map(int, row.Format.replace('x', '').split('*'))
                ans += row.Coefficient * xin[n1 - 1] * xin[n2 - 1]
            elif '^' in row.Format:
                n3, p1 = map(int, row.Format.replace('x', '').split('^'))
                ans += row.Coefficient * (xin[n3 - 1] ** p1)
            elif 'x' in row.Format:
                ans += row.Coefficient * xin[int(row.Format.replace('x', '')) - 1]
            else:
                ans += row.Coefficient
        return ans

    def show(self):
        if self.name:
            print 'Name: "%s"\n' % self.name
        print 'Format:\n%s\n' % self.format
        print 'coefficients:\n%s\n' % ', '.join([str(i) for i in self.b]) 
        print 'P Values:\n%s\n' % ', '.join([str(i) for i in self.pval])
        print 'R^2: %s' % str(self.r2)
        print '_' * 100


def fit(x, y, order=2, intercept=True, pair_terms=False, show=True,
        name='', return_ols=True):
    # make sure xi and y are of type np.ndarray
    for z in [x, y]:
        if not (isinstance(z, list) or isinstance(z, np.ndarray)):
            raise TypeError("Inputs must be a list or array")
    x = np.array(x)
    if len(x.shape) == 1:
        return ols_sing(x, y, order, intercept, show, name, return_ols)
    else:
        return ols_multi(x, y, order, intercept, pair_terms,
                         show, name, return_ols)


def ols(x, y, format='', show=True, name=''):
    # average y from data
    y_avg = y.mean()

    try:
        # Variance - Covariance Matrix: ([X]T * [X])^-1
        vcv = np.linalg.pinv(np.dot(x.T, x))
        # {b} = ([X]T * [X])^-1 * [X]T * {Y} * +/- {e}

        # {b} = [X]^-1 * {Y} * +/- {e}
        b = np.dot(np.linalg.pinv(x), y)
    except Exception as e:
        if show:
            print e
        return Solution('', np.array([]), np.array([]), 0, 0, str(e), np.array([]))

    # estimates
    y_est = np.array([np.dot(x[i, :], b) for i in xrange(len(x[:, 0]))])

    # residuals
    resid = y - y_est
    resid_sq = resid**2

    # mean square error
    mse = resid_sq.mean()

    # residual sum of squares
    resid_sos = resid_sq.sum()

    # Degrees of freedom, DOF = #Samples - #Terms
    tot_dof = len(x) - 1.
    reg_dof = x.shape[1]
    resid_dof = tot_dof - reg_dof
    dof = np.array([tot_dof, resid_dof, reg_dof])

    # Residual mean square (Variance)
    var = resid_sos / resid_dof

    # standard error (std deviation) of each coefficient = ((VCV matrix diagonal) * variance)^0.5
    std_err = (vcv.diagonal() * var)**0.5
    
    # abs(coefficient / standard error) = T-distribution statistic
    t_stat = abs(np.hstack(b) / std_err)

    # p-values based on T-distribution (EXCEL FUNCTION: TDIST(t_stat, DOF, 1)
    # T-dist significance signal
    pval = stats.t.sf(t_stat, resid_dof)*2

    # total sum of squares (all obs rel to the grand avg)
    tot_sos = ((y - y_avg)**2).sum()

    # regression sum of squares (tot sum of sq. - resid sum of sq.)
    reg_sos = tot_sos - resid_sos

    # sum of squares (all)
    #sos = np.array([tot_sos, resid_sos, reg_sos])

    # mean squares: (sum of sq.) / (DOF)
    #mean_sq = sos / dof

    # F: found by dividing two variances
    #resid_mean_sq = mean_sq[1]
    #reg_mean_sq = mean_sq[2]

    # F distribution, excel func: FDIST(mean_sq_REG / mean_sq_RESID, DOF_REG, DOF_RESID)
    #prob_f = 1 - stats.f.cdf(reg_mean_sq / resid_mean_sq, reg_dof, resid_dof)

    # correlation coefficient, R^2 = (tot_sos - resid_sos) / (tot_sos)
    r2 = reg_sos / tot_sos

    # Reverse format, b, & pval such that its order matches np.poly1d
    format = ', '.join(format.replace(',', '').split()[::-1])
    if len(b.shape) == 1:
        b = b[::-1]
    else:
        b = np.array([i[0] for i in b[::-1]])
    pval = pval[::-1]

    if show:
        print 'Name:\n%s\n' % name
        print 'Format:\n%s\n' % format
        print 'coefficients:\n%s\n' % ', '.join([str(i) for i in b]) 
        print 'P Values:\n%s\n' % ', '.join([str(i) for i in pval])
        print 'R^2: %s' % str(r2)
        print '_' * 100

    return Solution(format, b, pval, r2, mse, name, resid)


def ols_sing(x1, y, order=2, intercept=True, show=True, name='',
             return_ols=True):
    assert isinstance(x1, list) or isinstance(x1, np.ndarray)
    assert isinstance(y, list) or isinstance(y, np.ndarray)
    assert len(x1) == len(y)
    assert order >= 1
    
    x = np.array([[1]*len(x1), x1]).T if intercept else np.vstack(x1)
    y = np.array(y)
    
    format = 'b, x1' if intercept else 'x1'
    
    if order > 1:
        x1 = np.vstack(x1)
        for i in xrange(2, order+1):
            x = np.concatenate((x, x1**i), axis=1)
            format += ', x1^%i' % i

    assert x.shape[1] < len(x)-1, "More data or a lower order is needed to complete OLS"

    return ols(x, y, format, show, name) if return_ols else (x, format)


def ols_multi(xi, y, order=2, intercept=True, pair_terms=True, show=True,
              name='', return_ols=True):
    assert 1 <= order <= 2, "Only implemented to handle orders 1 and 2"

    # make sure xi and y are of type np.ndarray
    for z in [xi, y]:
        if not (isinstance(z, list) or isinstance(z, np.ndarray)):
            raise TypeError("Inputs must be a list or array")

    xi = np.array(xi)
    y = np.vstack(y)

    assert len(xi) == len(y), "Inputs and output must have the same length"

    if xi.shape[1] == 1:
        xi = np.vstack(xi)

    # initialize format
    format = ''
    xmade = False

    # add intercept
    if intercept:
        format += 'b '
        x = np.vstack([1]*len(xi))
        xmade = True

    # add in first order inputs to format & x
    format += ', '.join(['x' + str(i+1) for i in xrange(xi.shape[1])])
    if xmade:
        x = np.concatenate((x, xi), axis=1)
    else:
        x = xi.copy()

    # add second order terms
    if order == 2:
        format += ', ' + ', '.join(['x' + str(j+1) + '^2' for j in xrange(xi.shape[1])])
        x = np.concatenate((x, xi**2), axis=1)

    # add pair terms
    if pair_terms:
        for z1 in xrange(1, xi.shape[1]):
            for z2 in xrange(z1+1, xi.shape[1]+1):
                format += ', x%i*x%i' %(z1, z2)
                x = np.concatenate((x, np.vstack(x[:, z1] * x[:, z2])), axis=1)

    #assert x.shape[1] < len(x)-1, "More data or a lower order is needed to complete OLS"
    
    return ols(x, y, format, show, name) if return_ols else (x, format)

if __name__ == '__main__':
    x = np.random.random([500, 10])
    y = x.sum(axis=1)
    
    sol = ols_multi(x, y, 2)
    """
    order = 2
    intercept = True
    
    x = np.array([1, 2, 3, 4, 5])
    y = np.poly1d([-0.5, 4, 12])(x)
    
    sol = ols_sing(x, y, order, intercept, show=True)

    coef = sol[1]
    st = 0 if intercept else 1

    f = np.poly1d(coef)
    
    xn = np.linspace(0, x.max())
    yn = f(xn)
    
    plt.plot(x, y, 'x')
    plt.plot(xn, yn)
    plt.title('R2 = %0.4f' % sol[-1])
    plt.show()
    """