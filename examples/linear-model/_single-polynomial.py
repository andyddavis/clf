import sys 
sys.path.insert(0, '/home/add8536/Software/install/clf/lib/')

import numpy as np

import PyCoupledLocalFunctions as clf

para = clf.Parameters()
para.Add("InputDimension", 2)
para.Add("OutputDimension", 1)
para.Add("MaximumOrder", 5)
para.Add("LocalRadius", 1.0)

# create a total order multi-index set and the Legendre basis function
multiSet = clf.MultiIndexSet(para)
leg = clf.LegendrePolynomials()

# create the local function
center = np.zeros(2)
func = clf.LocalFunction(multiSet, leg, center, para)
