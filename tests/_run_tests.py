import unittest

# the path where the tests are
import sys
sys.path.insert(0, './tests')

# import the tests
from BasisFunctionsTests import *
from ModelTests import *
from SupportPointTests import *
from SupportPointCloudTests import *
from LocalFunctionsTests import *

class TestCLFInstall(unittest.TestCase):
    def test_load_clf(self):
        # load the coupled local function library---make sure there is no error
        import CoupledLocalFunctions as clf

if __name__=='__main__':
    unittest.main()
