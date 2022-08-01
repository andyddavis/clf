import sys 
sys.path.insert(0, '/home/add8536/Software/install/clf/lib/')

import numpy as np

import random

import PyCoupledLocalFunctions as clf

# import plotting packages and set default figure options
useserif = True # use a serif font with figures?
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, SymLogNorm
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

indim = 2 # the input dimension
outdim = 1 # the output dimension

para = clf.Parameters()
para.Add("InputDimension", indim)
para.Add("OutputDimension", outdim)
para.Add("MaximumOrder", 5)
para.Add("LocalRadius", 1.0)

# create a total order multi-index set and the Legendre basis function
multiSet = clf.MultiIndexSet(para)
leg = clf.LegendrePolynomials()

# create the local function
center = np.zeros(2)
func = clf.LocalFunction(multiSet, leg, center, para)

# create the identity model 
model = clf.IdentityModel(para)

###############################################
coeff = np.array([random.uniform(-1.0, 1.0) for i in range(func.NumCoefficients())])

n = int(50)
xvec = np.linspace(-1.0, 1.0, n)
fx = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        fx[i, j] = func.Evaluate([xvec[i], xvec[j]], coeff) [0]

X, Y = np.meshgrid(xvec, xvec)

fig = plt.figure()
ax = fig.add_subplot(111)
pc = ax.pcolor(X, Y, fx.T, cmap='plasma_r')
ax.plot(center[0], center[1], 'o', color='#252525')
cbar = plt.colorbar(pc)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel(r'$x_0$')
ax.set_ylabel(r'$x_1$')
plt.savefig('figures/fig_result.png', format='png')
plt.close(fig)
        
