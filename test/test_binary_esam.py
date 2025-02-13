import unittest
from binary_esam import BinaryESAM


class TestBinaryESAM(unittest.TestCase):

    def setUp(self) -> None:
        self.net_params = {'f': 1000,
                           'h': 8,
                           'f_h_sparsity': 0.1,
                           'h_f_sparsity_e': 1,
                           'h_f_sparsity_i': 0.5,
                           'e': 3,
                           'h_thresh': 9}

    # -----------------------------------------------------------------------------------------------------------------
    # test_initialisation_params
    # -----------------------------------------------------------------------------------------------------------------
    def test_init_good(self):
        esam = BinaryESAM(network_params=self.net_params)
        self.assertTrue(esam.initialised)

    def test_key_error(self):
        self.net_params = {'f': 10,
                           'h': 8,
                           'f_h_sparsity': 0.1,
                            # 'h_f_sparsity_e': 1,
                           'h_f_sparsity_i': 0.5,
                           'e': 3,
                           'h_thresh': 9}
        self.assertRaises(KeyError,
                          lambda: BinaryESAM(network_params=self.net_params))

    def test_value_error(self):
        self.net_params = {'f': 10,
                           'h': 8,
                           'f_h_sparsity': 0.1,
                            'h_f_sparsity_e': 2,
                           'h_f_sparsity_i': 0.5,
                           'e': 3,
                           'h_thresh': 9}
        self.assertRaises(ValueError,
                          lambda: BinaryESAM(network_params=self.net_params))


if __name__ == '__main__':
    unittest.main()
