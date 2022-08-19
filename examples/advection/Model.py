import numpy as np

import PyCoupledLocalFunctions as clf

class Model(clf.IdentityModel):
    def __init__(self, para):
        super().__init__(para)

    def RightHandSide(self, x):
        return  [np.cos(6.0*np.pi*x[0])*np.sin(3.0*np.pi*x[1])]
        
