import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import ols
import regreg
import neural
import itertools

"""
3.1 Plate appearances per team game to qualify
    PA = (AB + BB) / 162
Only AL & NL leagues
1955 - 2016
20 main parameters of interest
"""

# generic data paths
path_gen = 'Data\MLB_batting_stats.hdf'

# generic batting data
bat = pd.read_hdf(path_gen)

ns = np.arange(bat.G.max() + 1)
years = pd.unique(bat.yearID).tolist()

"""
t = ['avg', 'R', 'RBI', 'HR', 'SB', '2B', '3B']

def d(r):
    return r[t].sum() / r['SO']

bat['Rating'] = bat.apply(d, axis=1)

bat['Max'] = 0
for y in years:
    bat.loc[bat.yearID == y, 'Max'] = bat.loc[bat.yearID == y, 'Rating'].max()

bat['Rating'] /= bat['Max']

bat = bat.drop('Max', 1)
"""

def f(x, e):
    x = x.reshape([len(x)/2, 2])
    return e[0] * x[:, 0] * x[:, 1] + e[1] * x[:, 0]**2 + e[2] * x[:, 1] ** 2 + e[3] * x[:, 0] + e[4] * x[:, 1] + e[5]

bguess = np.ones(6)
lasso_x = bat[['HR', 'SO']].as_matrix().flatten()
lasso_y = bat['RBI'].as_matrix()

ridge_x = bat[['G', 'H']].as_matrix().flatten()
ridge_y = bat['AB'].as_matrix()

elastic_x = bat[['H', 'RBI']].as_matrix().flatten()
elastic_y = bat['HR'].as_matrix()

lasso = regreg.elastic_net(f, lasso_x, lasso_y, bguess, 1.0, 0.1, True)
lasso.alpha = 1
lasso.color = 'g'
lasso.typ = 'LASSO'
lasso.xn = 'HR'
lasso.yn = 'SO'
lasso.zn = 'RBI'

ridge = regreg.elastic_net(f, ridge_x, ridge_y, bguess, 0, 0.1, True)
ridge.alpha = 0
ridge.color = 'b'
ridge.typ = 'Ridge'
ridge.xn = 'G'
ridge.yn = 'H'
ridge.zn = 'AB'

elastic = regreg.elastic_net(f, elastic_x, elastic_y, bguess, 0.5, 0.1, True)
elastic.alpha = 0.5
elastic.color = 'c'
elastic.typ = 'Elastic'
elastic.xn = 'H'
elastic.yn = 'RBI'
elastic.zn = 'HR'

# surface plots
def srf(x, y, reg):
    x = x.reshape(len(x)/2, 2)
    x1 = x[:, 0]
    x2 = x[:, 1]
    X, Y = np.meshgrid(np.linspace(x1.min(), x1.max()), np.linspace(x2.min(), x2.max()))
    Z = np.array([f(np.array([i, j]), reg.x) for i,j in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='gray')
    ax.plot(x1, x2, y, '.', color=reg.color, alpha=0.25)
    ax.set_xlabel(reg.xn, fontsize=14)
    ax.set_ylabel(reg.yn, fontsize=14)
    ax.set_zlabel(reg.zn, fontsize=14)
    ax.set_title('Best Fit %s Regression\nalpha = %.1f' %(reg.typ, reg.alpha))
    fig.text(0.25, 0.75, 'R2 = %.3f' % reg.r_sq)
    
    return fig, ax


# parity plots
#"""
lasso_ymod = f(lasso_x, lasso.x)
ridge_ymod = f(ridge_x, ridge.x)
elastic_ymod = f(elastic_x, elastic.x)

lasso_r = range(20, 180, 20)
ridge_r = range(400, 800, 100)
elastic_r = range(0, 80, 10)


fig1, ax1 = plt.subplots()
ax1.set_title('HR, SO --> RBI; LASSO Regression\nParity Plot (alpha = %.1f)' % lasso.alpha)
ax1.plot(lasso_ymod, lasso_y, '.', color='g')
ax1.plot(lasso_r, lasso_r, color='black')
ax1.set_xlabel('Model RBI')
ax1.set_ylabel('Actual RBI')
ax1.set_xticks(lasso_r)
ax1.set_yticks(lasso_r)
fig1.text(0.25, 0.75, 'R2 = %.3f' % lasso.r_sq)

fig2, ax2 = plt.subplots()
ax2.set_title('G, H --> AB; Ridge Regression\nParity Plot (alpha = %.1f)' % ridge.alpha)
ax2.plot(ridge_ymod, ridge_y, '.', color='b')
ax2.plot(ridge_r, ridge_r, color='black')
ax2.set_xlabel('Model AB')
ax2.set_ylabel('Actual AB')
ax2.set_xticks(ridge_r)
ax2.set_yticks(ridge_r)
fig2.text(0.25, 0.75, 'R2 = %.3f' % ridge.r_sq)

fig3, ax3 = plt.subplots()
ax3.set_title('H, RBI --> HR; Elastic Net Regression\nParity Plot (alpha = %.1f)' % elastic.alpha)
ax3.plot(elastic_ymod, elastic_y, '.', color='c')
ax3.plot(elastic_r, elastic_r, color='black')
ax3.set_xlabel('Model HR')
ax3.set_ylabel('Actual HR')
ax3.set_xticks(elastic_r)
ax3.set_yticks(elastic_r)
fig3.text(0.25, 0.75, 'R2 = %.3f' % elastic.r_sq)

[f5.savefig('%s_Parity.png' % t, dpi=200) for f5, t in zip([fig1, fig2, fig3], ['LASSO', 'Ridge', 'Elastic'])]
#"""

# residuals histograms
#"""
lasso_resid = lasso_y - f(lasso_x, lasso.x)
#lasso_resid /= lasso_resid.max()

ridge_resid = ridge_y - f(ridge_x, ridge.x)
#ridge_resid /= ridge_resid.max()

elastic_resid = elastic_y - f(elastic_x, elastic.x)
#elastic_resid /= elastic_resid.max()

bins = 50.
lasso_hist, b1 = np.histogram(lasso_resid, bins=bins)
ridge_hist, b2 = np.histogram(ridge_resid, bins=bins)
elastic_hist, b3 = np.histogram(elastic_resid, bins=bins)

fig1, ax1 = plt.subplots()
ax1.set_title('HR, SO --> RBI residuals; LASSO Regression\nalpha = %.1f' % lasso.alpha)
ax1.bar(np.linspace(b1.min(), b1.max(), bins), lasso_hist, color='g', width=lasso_resid.max()/(bins+10))
ax1.set_xlabel('Residual')
ax1.set_ylabel('Frequency')
fig1.text(0.75, 0.75, 'R2 = %.3f' % lasso.r_sq)

fig2, ax2 = plt.subplots()
ax2.set_title('G, H --> AB residuals; Ridge Regression\nalpha = %.1f' % ridge.alpha)
ax2.bar(np.linspace(b2.min(), b2.max(), bins), ridge_hist, color='b', width=ridge_resid.max()/(bins+10))
ax2.set_xlabel('Residual')
ax2.set_ylabel('Frequency')
fig2.text(0.75, 0.75, 'R2 = %.3f' % ridge.r_sq)

fig3, ax3 = plt.subplots()
ax3.set_title('H, RBI --> HR residuals; Elastic Net Regression\nalpha = %.1f' % elastic.alpha)
ax3.bar(np.linspace(b3.min(), b3.max(), bins), elastic_hist, color='c', width=elastic_resid.max()/(bins+10))
ax3.set_xlabel('Residual')
ax3.set_ylabel('Frequency')
fig3.text(0.75, 0.75, 'R2 = %.3f' % elastic.r_sq)

fig1.savefig('LASSO_Resids.png', dpi=200)
fig2.savefig('Ridge_Resids.png', dpi=200)
fig3.savefig('Elastic_Resids.png', dpi=200)
#"""


if 0:
    # parameters of interest
    params = ['G',
              'AB',
              'R',
              'H',
              '2B',
              '3B',
              'HR',
              'RBI',
              'SB',
              'CS',
              'BB',
              'SO',
              'IBB',
              'HBP',
              'SH',
              'SF',
              'GIDP',
              'age',
              #'PA',
              'avg',
              'height',
              'weight'
              ]

    lasso = []
    ridge = []
    elastic = []
    for p in itertools.combinations(params, 3):
        if 'H' in p and 'AB' in p and 'avg' in p:
            continue
        for i in xrange(len(p)):
            y = p[i]
            x = list(p[:i] + p[i+1:])
            yval = bat[y].as_matrix()
            xval = bat[x].as_matrix().flatten()
            bguess = np.ones(6)
            s = regreg.elastic_net(f, xval, yval, bguess, 1.0, 0.1, True)
            s2 = regreg.elastic_net(f, xval, yval, bguess, 0, 0.1, True)
            s3 = regreg.elastic_net(f, xval, yval, bguess, 0.5, 0.1, True)
            #s2 = ols.ols_sing(xval, yval, order=2)
            #s2 = ols.ols_multi(xval, yval, pair_terms=True, intercept=True, show=False, name=', '.join(x) + ': ' + y)
            if s.success and s.r_sq >= 0:
                lasso.append((s, x, y, 'lasso'))
            if s2.success and s2.r_sq >= 0:
                ridge.append((s2, x, y, 'ridge'))
            if s3.success and s3.r_sq >= 0:
                elastic.append((s3, x, y, 'elastic'))
            #if s.r2 >= 0.75:
            #   sols.append(s)

    #sols.sort(key=lambda a: a.r2, reverse=True)
    #print '%i solutions were found with an R2 >= 0.8' % len(sols)

    ridge = sorted(ridge, key=lambda j: j[0].r_sq, reverse=True)
    lasso = sorted(lasso, key=lambda j: j[0].r_sq, reverse=True)
    elastic = sorted(elastic, key=lambda j: j[0].r_sq, reverse=True)

    yticks = np.arange(11) / 10.
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    ax1.bar(np.arange(len(ridge)), [r[0].r_sq for r in ridge], color='b')
    ax1.set_title('Ridge Regressions (alpha = 0)')
    ax1.set_ylabel('R2')
    ax1.set_yticks(yticks)
    ax1.set_xlabel('Regressions')

    ax2.bar(np.arange(len(elastic)), [e[0].r_sq for e in elastic], color='c')
    ax2.set_title('Elastic Net Regressions (alpha = 0.5)')
    ax2.set_ylabel('R2')
    ax2.set_yticks(yticks)
    ax2.set_xlabel('Regressions')

    ax3.bar(np.arange(len(lasso)), [o[0].r_sq for o in lasso], color='g')
    ax3.set_title('LASSO Regressions (alpha = 1)')
    ax3.set_ylabel('R2')
    ax3.set_yticks(yticks)
    ax3.set_xlabel('Regressions')

    hash = '\n' + '-' * 50 + '\n'
    func_desc = 'Function: (a * x1 * x2) + (b * x1^2) + (c * x2^2) + (d * x1) + (e * x2) + f'
    for n, k in zip(['Ridge', 'Elastic', 'LASSO'], [ridge, elastic, lasso]):
        with open('Data\{}_2in.txt'.format(n), 'w') as fid:
            for a in k:
                fid.write('{0}{1}'.format(func_desc, hash))
                fid.write('RegType: {}\n'.format(a[3]))
                fid.write('Inputs: {}\n'.format(', '.join(a[1])))
                fid.write('Output: {}\n'.format(a[2]))
                for k in a[0]:
                    fid.write('{0}: {1}\n'.format(k, str(a[0][k])))
                fid.write(hash)
