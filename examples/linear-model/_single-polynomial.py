import sys 
sys.path.insert(0, '/home/add8536/Software/install/clf/lib/')
sys.path.insert(0, '/home/andy/Software/install/clf/lib/')

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

class Model(clf.IdentityModel):
    def __init__(self, para):
        super().__init__(para)

    def RightHandSide(self, x):
        return [np.sin(3.0*np.pi*x[1])]
        
indim = 2 # the input dimension
outdim = 1 # the output dimension

para = clf.Parameters()
para.Add("InputDimension", indim)
para.Add("OutputDimension", outdim)
para.Add("MaximumOrder", 10)
para.Add("NumPoints", 500)

# create a total order multi-index set and the Legendre basis function
multiSet = clf.MultiIndexSet(para)
leg = clf.LegendrePolynomials()

# the center of the domain for the polynomial and the domain
center = np.zeros(2)
domain = clf.Hypercube(center-np.array([1.0]*indim), center+np.array([1.0]*indim))

# create the polynomial 
func = clf.LocalFunction(multiSet, leg, domain, para)

# create the identity model 
model = Model(para)

# create the local residual
resid = clf.LocalResidual(func, model, para);

# create the optimizer
lm = clf.DenseLevenbergMarquardt(clf.DenseCostFunction([resid]), para)

# compute the optimial coefficients
coeff = np.array([random.uniform(-1.0, 1.0) for i in range(func.NumCoefficients())])
status, cost, coeff, _ = lm.Minimize(coeff)
assert(status==clf.OptimizationConvergence.CONVERGED or
       status==clf.OptimizationConvergence.CONVERGED_FUNCTION_SMALL or
       status==clf.OptimizationConvergence.CONVERGED_GRADIENT_SMALL)

n = int(250)
xvec = np.linspace(-1.0, 1.0, n)
fx = np.zeros((n, n))
expected = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        pnt = [xvec[i], xvec[j]]
        fx[i, j] = func.Evaluate(pnt, coeff) [0]
        expected[i, j] = model.RightHandSide(pnt) [0]

X, Y = np.meshgrid(xvec, xvec)

fig = plt.figure()
ax = fig.add_subplot(111)
pc = ax.pcolor(X, Y, fx.T, cmap='plasma_r', vmin=-1.0, vmax=1.0)
ax.plot(center[0], center[1], 'o', color='#252525')
cbar = plt.colorbar(pc)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel(r'$x_0$')
ax.set_ylabel(r'$x_1$')
plt.savefig('figures/fig_single-polynomial-result.png', format='png')
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)
pc = ax.pcolor(X, Y, expected.T, cmap='plasma_r', vmin=-1.0, vmax=1.0)
ax.plot(center[0], center[1], 'o', color='#252525')
cbar = plt.colorbar(pc)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel(r'$x_0$')
ax.set_ylabel(r'$x_1$')
plt.savefig('figures/fig_sinple-polynomial-expected.png', format='png')
plt.close(fig)

