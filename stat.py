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


year = bat.groupby('yearID')

f = np.vectorize(lambda n, y: bat[(bat.G < n) & (bat.yearID == y)].count().playerID / float(len(bat[bat.yearID == y])))

ns = np.arange(bat.G.max() + 1)
years = range(bat.yearID.min(), bat.yearID.max()+1)

b = bat[bat.H > 100]

x1 = b.age.tolist()
x2 = b.G.tolist()
y = b.H.tolist()

plt.plot(x1, x2, '.')
plt.show()

#sol = ols.ols_multi(x1, x2, y)
"""
for y in years:
    plt.plot(ns, 100.*f(ns, y))
plt.xlabel('Max # of Games Played')
plt.ylabel('% of Players in League')
plt.show()
"""