try:
    from __futures__ import division
except:
    pass

# ragavi test files
# To run:
# export PYTHONPATH=/path/to/root/module
# export PYTHONPATH='/home/lexya/Documents/ragavi/ragavi'

import unittest
import numpy as np
import ragavi.ragavi as rg
from pyrap.tables import table

from future.utils import listitems, listvalues
from builtins import map


class TestRagavi(unittest.TestCase):

    x1 = np.random.randint(1, 30, (20))
    x2 = np.random.randint(1, 30, (20))
    fs = np.random.randint(0, 2, (20))
    fx1 = np.ma.masked_array(data=x1, mask=fs)
    fx2 = np.ma.masked_array(data=x2, mask=fs)
    inp = np.vectorize(complex)(fx1, fx2)

    def get_ampha(self, inp):
        amp = np.ma.sqrt(np.square(inp.real) + np.square(inp.imag))
        angle = np.rad2deg(np.ma.arctan(np.imag(inp) / np.real(inp)))
        return amp, 0, angle, 0

    def get_reim(self, inp):
        real = np.real(inp)
        im = np.imag(inp)
        return real, 0, im, 0

    def test_determine_table(self):
        input_data = ["whatever.G", "anything.B3", "ABcdEF.K023",
                      ".keija", "testy"]
        expected = ['.G', '.B', '.K', -1, -1]
        result = []
        for data in input_data:
            result.append(rg.determine_table(data))

        self.assertEqual(result, expected)
    """
    def test_name_2id(self):
        ms_name = "./files/1491291289.G0"
        ms = table(ms_name, ack=False)

        input_data = [ms, ('DEEP_2', 'NGC45-08', '0454-80', 'RANDOM-SOURCE'),
                      ]
        expected = [2, -1, -1, -1]
        result = []
        for data in input_data[1]:
            result.append(rg.name_2id(ms, data))

        ms.close()
        self.assertEqual(result, expected)
    """

    def test_data_prep_G(self):

        expected = [self.get_ampha(self.inp), self.get_reim(self.inp)]
        result = [rg.data_prep_G(self.inp, 0, 'ap'),
                  rg.data_prep_G(self.inp, 0, 'ri')]

        self.assertTrue(result, expected)

    def test_data_prep_B(self):

        expected = [self.get_ampha(self.inp), self.get_reim(self.inp)]
        result = [rg.data_prep_B(self.inp, 0, 'ap'),
                  rg.data_prep_B(self.inp, 0, 'ri')]

        self.assertTrue(result, expected)

    def test_data_prep_K(self):

        expected = [self.get_ampha(self.inp), self.get_ampha(self.inp)]
        result = [rg.data_prep_K(self.inp, 0, 'ap'),
                  rg.data_prep_K(self.inp, 0, 'ri')]

        self.assertTrue(result, expected)

    def test_data_prep_F(self):

        expected = [self.get_ampha(self.inp), self.get_reim(self.inp)]
        result = [rg.data_prep_F(self.inp, 0, 'ap'),
                  rg.data_prep_F(self.inp, 0, 'ri')]

        self.assertTrue(result, expected)


if __name__ == '__main__':
    unittest.main()
