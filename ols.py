import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def ols(x, y, format='', show=True):
    # average y from data
    y_avg = y.mean()

    # Variance - Covariance Matrix: ([X]T * [X])^-1
    vcv = np.linalg.pinv(np.dot(x.T, x))

    # {b} = ([X]T * [X])^-1 * [X]T * {Y} * +/- {e}
    b = np.dot(np.dot(vcv, x.T), y)

    # estimates
    y_est = np.array([np.dot(x[i, :], b) for i in xrange(len(x[:, 0]))])

    # residuals
    resid = y - y_est
    resid_sq = resid**2

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
    r_sq = reg_sos / tot_sos

    # Reverse format, b, & pval such that its order matches np.poly1d
    format = ', '.join(format.replace(',', '').split()[::-1])
    b = b[::-1]
    pval = pval[::-1]

    if show:
        print 'Format:\n%s\n' % format
        print 'coefficients:\n%s\n' % ', '.join([str(i) for i in b]) 
        print 'P Values:\n%s\n' % ', '.join([str(i) for i in pval])
        print 'R^2: %s' % str(r_sq)

    return format, b, pval, r_sq


def ols_sing(x1, y, order=2, intercept=True, show=True):
    assert isinstance(x1, list) or isinstance(x1, np.ndarray)
    assert isinstance(y, list) or isinstance(y, np.ndarray)
    assert len(x1) == len(y)
    assert order >= 1
    
    x = np.array([[1]*len(x1), x1]).T if intercept else np.vstack(np.array(x1))
    y = np.array(y)
    
    format = 'b, x' if intercept else 'x'
    
    if order > 1:
        x1 = np.vstack(np.array(x1))
        for i in xrange(2, order+1):
            x = np.concatenate((x, x1**i), axis=1)
            format += ', x^%i' % i

    assert x.shape[1] < len(x)-1, "More data or a lower order is needed to complete OLS"

    return ols(x, y, format, show)


def ols_multi(x1, x2, y, order=2, show=True):
    assert 1 <= order <= 2, "Only implemented to handle orders 1 and 2"
    
    format = "b, x1, x2"
    
    if order == 2:
        format += ", x1x2, x1^2, x2^2"
        
    for z in [x1, x2, y]:
        if not (isinstance(z, list) or isinstance(z, np.ndarray)):
            raise TypeError("Inputs must be a list or 1D array")
        if len(z) < order:
            raise ValueError("Not enough data points to solve OLS with order = %i") % order

    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.vstack(np.array(y))

    assert len(x1) == len(x2) == len(y), "Inputs and output must have the same length"

    # intercept
    yint = [1]*len(x1)

    # main x matrix
    x = np.array([yint, x1, x2]).T

    if order > 1:
        # (x1 * x2)
        x1x2 = x1 * x2

        # (x1)^2
        x1_sq = x1 ** 2
        
        # (x2)^2
        x2_sq = x2 ** 2
        
        h = np.array([x1x2, x1_sq, x2_sq]).T
        
        x = np.concatenate((x, h), axis=1)

    assert x.shape[1] < len(x)-1, "More data or a lower order is needed to complete OLS"
    
    return ols(x, y, format, show)

if __name__ == '__main__':
    order = 2
    intercept = True
    
    x = np.array([1, 2, 3, 4, 5])
    y = np.poly1d([-0.5, 4, 12])(x)
    
    sol = ols_sing(x, y, order, intercept, show=True)

    coef = sol[1]
    st = 0 if intercept else 1

    f = np.poly1d(coef[::-1])
    
    xn = np.linspace(0, x.max())
    yn = f(xn)
    
    plt.plot(x, y, 'x')
    plt.plot(xn, yn)
    plt.title('R2 = %0.4f' % sol[-1])
    plt.show()