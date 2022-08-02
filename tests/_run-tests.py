"""! @brief Run the python unit tests"""

import unittest 

import os, sys 
#sys.path.insert(0, '/home/add8536/Software/install/clf/lib/')
sys.path.insert(0, '/home/andy/Software/install/clf/lib/')
sys.path.insert(0, './tests')
#os.environ['LD_LIBRARY_PATH'] += ':/home/add8536/Software/install/clf/lib/'

from ParametersTests import *

from MultiIndexTests import *
from MultiIndexSetTests import *

from OrthogonalPolynomialsTests import *

from FeatureVectorTests import *
from FeatureMatrixTests import *

from LocalFunctionTests import *

from IdentityModelTests import *
from LinearModelTests import *

from PenaltyFunctionTests import *
from CostFunctionTests import *

from PointTests import *

from LocalResidualTests import *

class TestCLFInstall(unittest.TestCase):
    """! Test the python installation
    """
    def test_load_clf(self):
        """! Test module import
        """
        import PyCoupledLocalFunctions as clf

if __name__=='__main__':
    unittest.main()
