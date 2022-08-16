import sys 
sys.path.insert(0, '/home/add8536/Software/install/clf/lib/')
sys.path.insert(0, '/home/andy/Software/install/clf/lib/')

import numpy as np

import PyCoupledLocalFunctions as clf
from Model import *

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

Lx = 1.0 # x direction domain length
Ly = 1.0 # y direction domain length
nx = 10 # number of points in the x direction
ny = 3 # number of points in the y direction
dy = Ly/ny
dx = Lx/nx

para = clf.Parameters()
para.Add("InputDimension", indim) 
para.Add("OutputDimension", outdim)
para.Add("MaximumOrder", 10)
para.Add("NumPoints", 75)

# create a total order multi-index set and the Legendre basis function
multiSet = clf.MultiIndexSet(para)
leg = clf.LegendrePolynomials()

# create the identity model 
model = Model(para)

# create the global domain
globalDomain = clf.Hypercube(np.array([0.0, 0.0]), np.array([Lx, Ly]))

# create the point cloud
cloud = clf.PointCloud(globalDomain)
for i in range(nx):
    for j in range(ny):
        cloud.AddPoint(clf.Point([dx/2.0+i*dx, dy/2.0+j*dy]))
assert(cloud.NumPoints()==nx*ny)

domains = [None]*cloud.NumPoints()
funcs = [None]*cloud.NumPoints()
coeffs = [None]*cloud.NumPoints()
resids = [None]*cloud.NumPoints()
delta = np.array([dx/2.0, dy/2.0])
for ind in range(cloud.NumPoints()):
    # create the local domain 
    domains[ind] = clf.Hypercube(cloud.Get(ind).x-delta, cloud.Get(ind).x+delta)

    # create the local polynomial
    funcs[ind] = clf.LocalFunction(multiSet, leg, domains[ind], para)
    
    # create the local residual
    resids[ind] = clf.LocalResidual(funcs[ind], model, para)
    
    # create the optimizer
    lm = clf.DenseLevenbergMarquardt(clf.DenseCostFunction(resids[ind]), para)
    
    # compute the optimial coefficients
    coeffs[ind] = np.array([0.0]*funcs[ind].NumCoefficients())
    status, cost, coeffs[ind], _ = lm.Minimize(coeffs[ind])
    assert(status==clf.OptimizationConvergence.CONVERGED or
           status==clf.OptimizationConvergence.CONVERGED_FUNCTION_SMALL or
           status==clf.OptimizationConvergence.CONVERGED_GRADIENT_SMALL)

n = int(250)
xvec = np.linspace(0, Lx, n)
yvec = np.linspace(0, Ly, n)
fx = np.zeros((n, n))
expected = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        pnt = [xvec[i], yvec[j]]
        ind, _ = cloud.ClosestPoint(pnt)
        fx[i, j] = funcs[ind].Evaluate(pnt, coeffs[ind]) [0]
        expected[i, j] = model.RightHandSide(pnt) [0]
        
X, Y = np.meshgrid(xvec, yvec)

fig = plt.figure()
ax = fig.add_subplot(111)
pc = ax.pcolor(X, Y, fx.T, cmap='plasma_r', vmin=-1.0, vmax=1.0)
for i in range(cloud.NumPoints()):
    ax.plot(cloud.Get(i).x[0], cloud.Get(i).x[1], 'o', color='#252525', markersize=5)
    for j in range(resids[i].NumLocalPoints()):
        ax.plot(resids[i].GetPoint(j).x[0], resids[i].GetPoint(j).x[1], 's', color='#252525', markersize=2)
cbar = plt.colorbar(pc)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel(r'$x_0$')
ax.set_ylabel(r'$x_1$')
plt.savefig('figures/fig_multiple-polynomials-grid-result.png', format='png')
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)
pc = ax.pcolor(X, Y, expected.T, cmap='plasma_r', vmin=-1.0, vmax=1.0)
cbar = plt.colorbar(pc)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel(r'$x_0$')
ax.set_ylabel(r'$x_1$')
plt.savefig('figures/fig_multiple-polynomials-grid-expected.png', format='png')
plt.close(fig)
