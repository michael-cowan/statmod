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

def f(x, e):
    x = x.reshape([len(x)/2, 2])
    return e[0] * x[:, 0] * x[:, 1] + e[1] * x[:, 0]**2 + e[2] * x[:, 1] ** 2 + e[3] * x[:, 0] + e[4] * x[:, 1] + e[5]

"""
inp = bat[['H', 'R']].as_matrix().flatten()
outp = bat.HR.as_matrix()
bguess = np.ones(4)
s = regreg.elastic_net(f, inp, outp, bguess, 1., 0.1, False)
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

    sols = []
    for p in itertools.combinations(params, 3):
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
            if s.success:
                sols.append((s, x, y, 'lasso'))
            if s2.success:
                sols.append((s2, x, y, 'ridge'))
            if s3.success:
                sols.append((s3, x, y, 'elastic'))
            #if s.r2 >= 0.75:
            #   sols.append(s)

    #sols.sort(key=lambda a: a.r2, reverse=True)
    #print '%i solutions were found with an R2 >= 0.8' % len(sols)
for a in sols:
    with open('Data\LinRegReg_2in.txt', 'a') as fid:
        fid.write('RegType: {}\n'.format(a[3]))
        fid.write('Inputs: {}\n'.format(', '.join(a[1])))
        fid.write('Output: {}\n'.format(a[2]))
        for k in a[0]:
            fid.write('{0}: {1}\n'.format(k, str(a[0][k])))
        fid.write('\n{}\n'.format('-'*50))