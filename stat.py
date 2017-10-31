import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ols
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
    for p in itertools.combinations(params, 4):
        for i in xrange(len(p)):
            y = p[i]
            x = list(p[:i] + p[i+1:])
            yval = bat[y].tolist()
            xval = bat[x].as_matrix()
        
            s = ols.ols_multi(xval, yval, pair_terms=True, intercept=True, show=False, name=', '.join(x) + ': ' + y)
            if s.r2 >= 0.8:
                sols.append(s)

    sols.sort(key=lambda a: a.r2, reverse=True)
    print '%i solutions were found with an R2 >= 0.8' % len(sols)