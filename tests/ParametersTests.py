import unittest

import PyCoupledLocalFunctions as clf

class TestParameters(unittest.TestCase):
    def test_basic_test(self):
        para = clf.Parameters()
        self.assertEqual(para.NumParameters(), 0)

        para.Add('A', 1.4)
        self.assertEqual(para.NumParameters(), 1)
        self.assertAlmostEqual(para.Get('A'), 1.4)
        self.assertAlmostEqual(para.Get('A', 2), 1.4)

        para.Add('B', 3)
        self.assertEqual(para.NumParameters(), 2)
        self.assertAlmostEqual(para.Get('B'), 3)
        self.assertAlmostEqual(para.Get('B', 2), 3)

        para.Add('C', 'test')
        self.assertEqual(para.NumParameters(), 3)
        self.assertEqual(para.Get('C'), 'test')
        self.assertEqual(para.Get('C', 3), 'test')

        self.assertEqual(para.Get('D', 'test again'), 'test again')

