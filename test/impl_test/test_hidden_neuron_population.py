import unittest
import numpy as np

from impl.connections import ConnectionType
from impl.hidden_neuron_population import HiddenNeurons

class TestHiddenNeurons(unittest.TestCase):

    def setUp(self):
        self.f = 5
        self.h = 2

    def test_init(self):
        hidden_neurons = HiddenNeurons(f=self.f,
                                       h=self.h,
                                       conn_f_h=1,
                                       conn_h_f_e=1,
                                       conn_h_f_i=1,
                                       f_h_conn_type=ConnectionType.FIXED_PROBABILITY,
                                       h_f_conn_type=ConnectionType.FIXED_PROBABILITY,
                                       h_threshold=1)
        memories = np.array([[1,1,1,0,0],[1,1,0,0,0],[0,0,0,1,1]])
        attract_points = hidden_neurons.mature_hidden_for_memories(memories)
        self.assertTrue(np.array_equal(memories, attract_points))

    def test_conns(self):
        hidden_neurons = HiddenNeurons(f=self.f,
                                       h=self.h,
                                       conn_f_h=1,
                                       conn_h_f_e=1,
                                       conn_h_f_i=1,
                                       f_h_conn_type=ConnectionType.FIXED_PROBABILITY,
                                       h_f_conn_type=ConnectionType.FIXED_PROBABILITY,
                                       h_threshold=1)
        memories = np.array([[1, 0, 1, 0, 0],
                             [1, 0, 0, 0, 0]])
        hidden_neurons.mature_hidden_for_memories(memories)
        expected_f_h = np.array([[1, 0, 1, 0, 0],
                                 [1, 0, 1, 0, 0],
                                 [1, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0]])
        expected_h_f = np.array([[1, -1, 1, -1, -1],
                                 [1, -1, 1, -1, -1],
                                 [1, -1, -1, -1, -1],
                                 [1, -1, -1, -1, -1]])
        self.assertTrue(np.array_equal(expected_h_f, hidden_neurons.h_f_conns))
        self.assertTrue(np.array_equal(expected_f_h, hidden_neurons.f_h_conns))

    def test_conns_2(self):
        hidden_neurons = HiddenNeurons(f=self.f,
                                       h=self.h,
                                       conn_f_h=0,
                                       conn_h_f_e=1,
                                       conn_h_f_i=0,
                                       f_h_conn_type=ConnectionType.FIXED_PROBABILITY,
                                       h_f_conn_type=ConnectionType.FIXED_PROBABILITY,
                                       h_threshold=1)
        memories = np.array([[1, 0, 1, 0, 0],
                             [1, 0, 0, 0, 0]])
        hidden_neurons.mature_hidden_for_memories(memories)
        expected_f_h = np.array([[0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0]])
        expected_h_f = np.array([[1, 0, 1, 0, 0],
                                 [1, 0, 1, 0, 0],
                                 [1, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0]])

        self.assertTrue(np.array_equal(expected_h_f, hidden_neurons.h_f_conns))
        self.assertTrue(np.array_equal(expected_f_h, hidden_neurons.f_h_conns))



if __name__ == '__main__':
    unittest.main()
