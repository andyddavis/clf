import numpy as np

import h5py as h5

import os

import itertools

# import plotting packages and set default figure options
useserif = True # use a serif font with figures?
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
if useserif:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['text.usetex'] = True
else:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

filename = 'Poisson-1d.h5'
file = h5.File(filename, 'r')

supportPoints = file['/support points'] [()]
sortedIndices = sorted(range(len(supportPoints)), key=lambda k: supportPoints[k][0])
collocationPoints = list()
for i in range(len(supportPoints)):
    collocationPoints.append(file['/collocation points/support point '+str(i)] [()])

fig = plt.figure()
ax = fig.add_subplot(111)
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']
for ind, c in zip(sortedIndices, itertools.cycle(colors)):
    for y in collocationPoints[ind]:
        ax.plot(y, [0.0], 'x', color=c)
    ax.plot(supportPoints[ind], [0.0], 'o', color=c)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'')
plt.savefig('figures/fig_points.png', format='png')
plt.close(fig)
