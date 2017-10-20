import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ols

"""
3.1 Plate appearances per team game to qualify
"""

# generic data paths
path_gen = 'Data\MLB_{}_stats.csv'

# generic batting & fielding data
bat = pd.read_csv(path_gen.format('batting'))
field = pd.read_csv(path_gen.format('fielding'))


##### REFINING DATA #####
# plate appearances (PA)
bat['PA'] = bat.apply(lambda r: (r.AB + r.BB) / 162., axis=1)

# PA must be > 3.1 to qualify for batting title
bat = bat[bat.PA >= 3.1]


year = bat.groupby('yearID')
oplayer = bat.groupby('playerID')
dplayer = field.groupby('playerID')

f = np.vectorize(lambda n, y: bat[(bat.G < n) & (bat.yearID == y)].count().playerID / float(len(bat[bat.yearID == y])))

ns = np.arange(bat.G.max() + 1)
years = range(bat.yearID.min(), bat.yearID.max()+1)

b = bat[bat.H != 0].sort('G')



x1 = b.age.tolist()
x2 = b.G.tolist()
y = b.H.tolist()

#sol = ols.ols_multi(x1, x2, y)
sol = ols.ols_sing(x2, y, order=2)

plt.plot(x2, y, '.', alpha=0.7)
plt.plot(x2, np.poly1d(sol.b)(x2))
plt.legend(['Data', 'Model'])
plt.xlabel('Games played')
plt.ylabel('Number of hits')
plt.show()
"""
for y in years:
    plt.plot(ns, 100.*f(ns, y))
plt.xlabel('Max # of Games Played')
plt.ylabel('% of Players in League')
plt.show()
"""