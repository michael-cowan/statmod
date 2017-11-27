import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ols
import regreg
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


d = bat[bat.playerID == 'ortizda01']

plt.plot(d.age, d.Rating)
plt.show()

def f(x, e):
    return e[0] * np.cos(np.pi * e[1] * x) + e[2] * np.sin(np.pi * e[3] * x)


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

    sols = []
    for p in itertools.combinations(params, 2):
        for i in xrange(len(p)):
            y = p[i]
            x = list(p[:i] + p[i+1:])
            #yval = bat[y].as_matrix()
            #xval = bat[x].as_matrix()
            yval = bat['weight'].as_matrix()
            xval = bat['height'].as_matrix()
            bguess = np.ones(3)
            s = regreg.elastic_net(regreg.polyfunc, xval, yval, bguess, 0.5, 0.1, False)
            s2 = ols.ols_sing(xval, yval, order=2)
            #s2 = ols.ols_multi(xval, yval, pair_terms=True, intercept=True, show=False, name=', '.join(x) + ': ' + y)
            break
            #if s.r2 >= 0.75:
            #   sols.append(s)
        break

    #sols.sort(key=lambda a: a.r2, reverse=True)
    #print '%i solutions were found with an R2 >= 0.8' % len(sols)