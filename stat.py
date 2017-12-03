import numpy as np
import matplotlib.pyplot as plt
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

if 1:
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
    for p in itertools.combinations(params, 2):
        if 'H' in p and 'AB' in p and 'avg' in p:
            continue
        for i in [1]:#xrange(len(p)):
            x, y = p
            yval = bat[y].as_matrix()
            xval = bat[x].as_matrix()
            #s = regreg.elastic_net(f, xval, yval, bguess, 1.0, 0.1, True)
            #s2 = regreg.elastic_net(f, xval, yval, bguess, 0, 0.1, True)
            #s3 = regreg.elastic_net(f, xval, yval, bguess, 0.5, 0.1, True)
            s2 = ols.ols_sing(xval, yval, order=2, intercept=True, show=False, name=x + ': ' + y)
            #s2 = ols.ols_multi(xval, yval, order=2, pair_terms=True, intercept=True, show=False, name=', '.join(x) + ': ' + y)
            if s2.r2 >= 0:
               single.append(s2)
    single = sorted(single, key=lambda x: x.r2, reverse=True)

    fig, ax = plt.subplots()
    ax.bar(range(len(single)), [i.r2 for i in single], color='r')
    ax.set_title('Single Variable OLS\n(a * x^2) + (b * x) + intercept')
    ax.set_xlabel('Regressions')
    ax.set_ylabel('R2')

    fig.savefig('SingleOLS_Reg1.png', dpi=200)


    a = single[0]
    vals = bat[['HR', 'RBI']].sort_values('HR').as_matrix()
    x = vals[:, 0]
    y = vals[:, 1]
    h, b = np.histogram(a.resid, bins=50.)


    fig2, ax2 = plt.subplots()
    ax2.bar(np.linspace(0, a.resid.max(), 50), h, color='r', width=a.resid.max()/60.)
    ax2.set_title('HR --> RBI residuals; OLS')
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Frequency')
    fig2.text(0.75, 0.75, 'R2 = %.3f' % a.r2)

    fig2.savefig('SingleOLS_Resids1.png', dpi=200)


    yhat = np.poly1d(a.b)(x)
    fig3, ax3 = plt.subplots()
    ax3.plot(yhat, y, '.', color='r')
    ax3.set_title('HR --> RBI; OLS\nParity Plot')
    ax3.set_xlabel('Model RBI')
    ax3.set_ylabel('Actual RBI')
    ax3.set_xticks(range(20, 180, 20))
    ax3.set_yticks(range(20, 180, 20))
    fig3.text(0.25, 0.75, 'R2 = %.3f' % a.r2)

    fig3.savefig('SingleOLS_Parity1.png', dpi=200)


    fig4, ax4 = plt.subplots()
    ax4.plot(x, y, '.', color='r')
    ax4.set_title('Best Single Variable OLS Fit')
    ax4.set_xlabel('HR')
    ax4.set_ylabel('RBI')
    ax4.plot(x, yhat, color='black')
    ax4.legend(['Data', 'OLS Fit'])
    fig4.text(0.75, 0.25, 'R2 = %.3f' % a.r2)

    fig4.savefig('SingleOLS_Plot1.png', dpi=200)