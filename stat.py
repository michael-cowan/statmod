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
    return e[0] * x[:, 0] * x[:, 1] + e[1] * x[:, 0]**2 + e[2] * x[:, 1] ** 2 + e[3] * x[:, 0] + e[4] * x[:, 1] + e[5]

x = ['G', 'avg']
y = 'H'
xn = 'G, avg'
yn = 'H'
tot = bat[['G', 'avg', 'H']].sort_values('avg')
xval = tot[['G', 'avg']].as_matrix()
yval = tot.H.as_matrix()

s2 = ols.ols_multi(xval, yval, order=2, pair_terms=True, intercept=True, show=False, name=', '.join(x) + ': ' + y)

fig, ax = plt.subplots()

ax.plot([s2.func(i) for i in xval], yval, '.', color='m')
ax.set_xticks(range(70, 260, 20))
ax.set_yticks(range(70, 260, 20))
ax.plot(range(70, 251), range(70, 251), color='black')
ax.set_xlabel('Model')
ax.set_ylabel('Actual')
fig.text(0.25, 0.75, 'R2 = %.3f' % s2.r2)
ax.set_title('Multivariate OLS\nParity Plot')


"""
x1 = np.linspace(xval[:, 0].min(), xval[:, 0].max())
x2 = np.linspace(xval[:, 1].min(), xval[:, 1].max())
X, Y = np.meshgrid(x1, x2)
Z = np.array([f(y1, y2) for y1,y2 in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, color='gray')
ax.plot(xval[:, 0], xval[:, 1], yval, '.', color='m', alpha=0.25)
ax.set_xlabel('G', fontsize=14)
ax.set_ylabel('avg', fontsize=14)
ax.set_zlabel('H', fontsize=14)
ax.set_title('Best Fit Multivariate OLS')
fig.text(0.25, 0.75, 'R2 = %.3f' % s2.r2)
"""

def make_str(o):
    n = 97
    s = ''
    for i in xrange(o, 0, -1):
        s += ' + (%s * x^%i)' %(chr(n), i)
        n += 1
    s += ' + %s' %(chr(n))
    return 'y = ' + s.replace('^1', '').strip(' + ').strip()



colors = ['m', 'orange', 'r', 'black']
o = 0
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

    single = []
    for p in itertools.combinations(params, 3):
        if 'H' in p and 'AB' in p and 'avg' in p:
            continue
        for i in xrange(len(p)):
            y = p[i]
            x = list(p[:i] + p[i+1:])
            yval = bat[y].as_matrix()
            xval = bat[x].as_matrix()
            #s = regreg.elastic_net(f, xval, yval, bguess, 1.0, 0.1, True)
            #s2 = regreg.elastic_net(f, xval, yval, bguess, 0, 0.1, True)
            #s3 = regreg.elastic_net(f, xval, yval, bguess, 0.5, 0.1, True)
            #s2 = ols.ols_sing(xval, yval, order=o, intercept=True, show=False, name=x + ': ' + y)
            s2 = ols.ols_multi(xval, yval, order=2, pair_terms=True, intercept=True, show=False, name=', '.join(x) + ': ' + y)
            if s2.r2 >= 0:
               single.append(s2)
    single = sorted(single, key=lambda x: x.r2, reverse=True)

    fig, ax = plt.subplots()
    ax.bar(range(len(single)), [i.r2 for i in single], color=colors[o])
    ax.set_title('Multivariate OLS\ny = (a * x2 * x1) + (b * x2^2) + (c * x1^2) + (d * x2) + (e * x1) + f')
    ax.set_xlabel('Regressions')
    ax.set_ylabel('R2')

    fig.savefig('MultiOLS_Reg.png', dpi=200)


    a = single[0]
    xn,yn = a.name.split(': ')
    if 0:
        vals = bat[[xn, yn]].sort_values(xn).as_matrix()
        x = vals[:, 0]
        y = vals[:, 1]
        h, b = np.histogram(a.resid, bins=50.)

        fig2, ax2 = plt.subplots()
        ax2.bar(np.linspace(0, a.resid.max(), 50), h, color=colors[o], width=a.resid.max()/60.)
        ax2.set_title('%s --> %s residuals; OLS' %(xn, yn))
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Frequency')
        fig2.text(0.75, 0.75, 'R2 = %.3f' % a.r2)

        fig2.savefig('SingleOLS_Resids%i.png' % o, dpi=200)


        yhat = np.poly1d(a.b)(x)
        fig3, ax3 = plt.subplots()
        ax3.plot(yhat, y, '.', color=colors[o])
        ax3.set_title('%s --> %s; OLS\nParity Plot' %(xn, yn))
        ax3.set_xlabel('Model %s' % yn)
        ax3.set_ylabel('Actual %s' % yn)
        ax3.set_xticks(range(20, 180, 20))
        ax3.set_yticks(range(20, 180, 20))
        fig3.text(0.25, 0.75, 'R2 = %.3f' % a.r2)

        fig3.savefig('SingleOLS_Parity%i.png' % o, dpi=200)


        fig4, ax4 = plt.subplots()
        ax4.plot(x, y, '.', color=colors[o])
        ax4.set_title('Best Single Variable OLS Fit')
        ax4.set_xlabel(xn)
        ax4.set_ylabel(yn)
        ax4.plot(x, yhat, color='black')
        ax4.legend(['Data', 'OLS Fit'])
        fig4.text(0.75, 0.25, 'R2 = %.3f' % a.r2)

        fig4.savefig('SingleOLS_Plot%i.png' % o, dpi=200)