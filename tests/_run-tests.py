import unittest 

import os, sys 
sys.path.insert(0, '/home/add8536/Software/install/clf/lib/')
#os.environ['LD_LIBRARY_PATH'] += ':/home/add8536/Software/install/clf/lib/'

class TestCLFInstall(unittest.TestCase):
    def test_load_clf(self):
        import PyCoupledLocalFunctions as clf

class TestParameters(unittest.TestCase):
    def test_basic_test(self):
        import PyCoupledLocalFunctions as clf
        para = clf.Parameters()
        print(para.NumParameters())
        para.Add("A", 3.2)
        print(para.NumParameters())
        print("A:", para.Get("A"))

if __name__=='__main__':
    unittest.main()
