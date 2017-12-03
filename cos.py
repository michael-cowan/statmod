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

def make_df(prev=3):
    cols = range(1, prev + 1) + ['Out']
    df = pd.DataFrame(columns = cols)
    for p in bat.playerID.unique():
        d = bat.loc[bat.playerID == p, ['yearID', 'Rating']]
        if len(d) < prev+1:
            continue
        d.index = range(len(d))
        jumps = np.where((np.diff(d.yearID) >= 2))[0]
        jumps = [j for j in jumps if d.yearID.tolist()[j] != 1980]
        splits = (d.iloc[:j+1] for j in jumps)
        for s in splits:
            if len(s) >= prev+1:
                for k in xrange(len(s.index[:-prev])):
                    inds = s.index.tolist()[k:k+prev+1]
                    vals = s.loc[inds, 'Rating']
                    row = {m: n for m,n in zip(cols, vals)}
                    df = df.append(row, ignore_index=True)
    return df

for p in xrange(1, 11):
    df = make_df(p)
    df.to_csv('Data\COS_prev%i.csv' % p, index=False)