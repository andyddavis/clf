import sys 
sys.path.insert(0, '/home/add8536/Software/install/clf/lib/')

import PyCoupledLocalFunctions as clf

para = clf.Parameters()
para.Add("InputDimension", 2)
para.Add("MaximumOrder", 5)

# create a total order multi-index set
multiSet = clf.MultiIndexSet(para)
