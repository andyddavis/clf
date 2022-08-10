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
        return [np.sin(2.0*np.pi*x[1])]

indim = 2 # the input dimension
outdim = 1 # the output dimension

Lx = 2.0 # x direction domain length
Ly = 1.0 # y direction domain length
nx = 2 # number of points in the x direction
ny = 2 # number of points in the y direction
dy = Ly/ny
dx = Lx/nx

para = clf.Parameters()
para.Add("InputDimension", indim) 
para.Add("OutputDimension", outdim)
para.Add("MaximumOrder", 5)
para.Add("LocalRadius", 0.5)
para.Add("NumPoints", 500)

# create a total order multi-index set and the Legendre basis function
multiSet = clf.MultiIndexSet(para)
leg = clf.LegendrePolynomials()

# create the point cloud
cloud = clf.PointCloud()
for i in range(nx):
    for j in range(ny):
        cloud.AddPoint([dx/2.0+i*dx, dy/2.0+j*dy])

# create the coupled local function
func = clf.CoupledLocalFunction(cloud)

# the center of the domain for the polynomial
#centers = [clf.Point(np.array([-0.5, 0.5])), clf.Point(np.array([0.5, 0.5])), clf.Point(np.array([0.5, -0.5])), clf.Point(np.array([-0.5, -0.5]))]           

# create the local polynomials
#funcs = [clf.LocalFunction(multiSet, leg, center, para) for center in centers]

"""
# create the identity model 
model = Model(para)

# create the local residual
resid = clf.LocalResidual(func, model, center, para);

# create the optimizer
lm = clf.DenseLevenbergMarquardt(clf.DenseCostFunction([resid]), para)

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
        fx[i, j] = func.Evaluate([xvec[i], xvec[j]], coeff) [0]
        expected[i, j] = model.RightHandSide([xvec[i], xvec[j]]) [0]

X, Y = np.meshgrid(xvec, xvec)

fig = plt.figure()
ax = fig.add_subplot(111)
pc = ax.pcolor(X, Y, fx.T, cmap='plasma_r', vmin=-1.0, vmax=1.0)
ax.plot(center.x[0], center.x[1], 'o', color='#252525')
cbar = plt.colorbar(pc)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel(r'$x_0$')
ax.set_ylabel(r'$x_1$')
plt.savefig('figures/fig_result.png', format='png')
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)
pc = ax.pcolor(X, Y, expected.T, cmap='plasma_r', vmin=-1.0, vmax=1.0)
ax.plot(center.x[0], center.x[1], 'o', color='#252525')
cbar = plt.colorbar(pc)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel(r'$x_0$')
ax.set_ylabel(r'$x_1$')
plt.savefig('figures/fig_expected.png', format='png')
plt.close(fig)

"""
