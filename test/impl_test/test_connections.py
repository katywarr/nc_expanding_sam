import unittest
import numpy as np
from impl.connections import get_sparse_connection_matrix, ConnectionType

class TestConnections(unittest.TestCase):

    # -----------------------------------------------------------------------------------------------------------------
    # test_get_sparse_connections
    # -----------------------------------------------------------------------------------------------------------------
    def test_get_sparse_connections_sparsity_e_1(self):
        # Test connections with probability = 1
        mem = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        h_per_mem = 9
        sparsity_e = 1.0

        connections = get_sparse_connection_matrix(memory=mem, h=h_per_mem, sparsity_e=sparsity_e)

        # As probability is 1 the values should not have changed
        self.assertTrue(np.array_equal(mem, connections[0]))
        # Also check the shape of the return
        self.assertTrue(np.array_equal(mem.shape[0], connections.shape[1]))
        self.assertTrue(np.array_equal(h_per_mem, connections.shape[0]))

    def test_get_sparse_connections_sparsity_e(self):
        mem = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        h_per_mem = 2
        sparsity_e = 0.4
        np.random.seed(1)
        connections = get_sparse_connection_matrix(memory=mem, h=h_per_mem, sparsity_e=sparsity_e)

        expected = np.array([[0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 1, 0, 0, 0, 0, 0]])

        self.assertTrue(np.array_equal(expected, connections))
        # Also check the shape of the return
        self.assertTrue(np.array_equal(mem.shape[0], connections.shape[1]))
        self.assertTrue(np.array_equal(h_per_mem, connections.shape[0]))

    def test_get_sparse_connections_sparsity_e_no_i(self):
        # Check setting the inhibitory probability doesn't affect anything if there are no -1 values
        mem = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        h_per_mem = 2
        sparsity_e = 0.4
        sparsity_i = 0.1

        np.random.seed(1)

        connections = get_sparse_connection_matrix(memory=mem, h=h_per_mem, sparsity_e=sparsity_e, sparsity_i=sparsity_i)
        expected = np.array([[0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 1, 0, 0, 0, 0, 0]])

        self.assertTrue(np.array_equal(expected, connections))
        # Also check the shape of the return
        self.assertTrue(np.array_equal(mem.shape[0], connections.shape[1]))
        self.assertTrue(np.array_equal(h_per_mem, connections.shape[0]))

    def test_get_sparse_connections_sparsity_e_and_i(self):
        polar_binary_mem = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
        h_per_mem = 2
        sparsity_e = 0.4
        sparsity_i = 0.5

        np.random.seed(2)

        connections = get_sparse_connection_matrix(memory=polar_binary_mem, h=h_per_mem,
                                                   sparsity_e=sparsity_e, sparsity_i=sparsity_i)
        expected = np.array([[0, 1, 0, 0, 0, -1, -1, 0, -1, -1],
                             [0, 0, 1, 0, 1, 0, 0, -1, 0, -1]])

        self.assertTrue(np.array_equal(expected, connections))
        # Also check the shape of the return
        self.assertTrue(np.array_equal(polar_binary_mem.shape[0], connections.shape[1]))
        self.assertTrue(np.array_equal(h_per_mem, connections.shape[0]))

    def test_get_sparse_connections_sparsity_e_and_i_columns_1(self):
        polar_binary_mem = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
        h_per_mem = 2
        sparsity_e = 0.4
        sparsity_i = 0.5
        np.random.seed(2)

        connections = get_sparse_connection_matrix(memory=polar_binary_mem, h=h_per_mem,
                                                   sparsity_e=sparsity_e, sparsity_i=sparsity_i)
        expected = np.array([[0, 1, 0, 0, 0, -1, -1, 0, -1, -1],
                             [0, 0, 1, 0, 1, 0, 0, -1, 0, -1]])

        self.assertTrue(np.array_equal(expected, connections))
        # Also check the shape of the return
        self.assertTrue(np.array_equal(polar_binary_mem.shape[0], connections.shape[1]))
        self.assertTrue(np.array_equal(h_per_mem, connections.shape[0]))

    def test_get_sparse_connections_proportion_e_and_i(self):
        mem = np.array([1, 1, 1, 1, 1, -1, -1, 0, -1, -1])
        h_per_mem = 5
        sparsity_e = 0.2
        sparsity_i = 0.5

        np.random.seed(2)

        connections = get_sparse_connection_matrix(memory=mem, h=h_per_mem,
                                                   sparsity_e=sparsity_e, sparsity_i=sparsity_i,
                                                   connection_type=ConnectionType.FIXED_NUMBER)

        for row in connections:
            # There should always be exactly 1 one and 2 -1s

            self.assertEqual(1, np.where(row == 1)[0].shape[0])
            self.assertEqual(2, np.where(row == -1)[0].shape[0])

        # Check the shape of the return
        self.assertTrue(np.array_equal(mem.shape[0], connections.shape[1]))
        self.assertTrue(np.array_equal(h_per_mem, connections.shape[0]))

    def test_get_sparse_connections_proportion_e_fixed_pre(self):
        mem = np.array([1, 1, 1, 1, 1, 0, 0, -1, 0, -1, 0, 0])
        h_per_mem = 24
        sparsity_e = 0.8
        sparsity_i = 0.5

        np.random.seed(2)

        # Static numbers per h:  e: 4    i: 1
        connections = get_sparse_connection_matrix(memory=mem, h=h_per_mem,
                                                   sparsity_e=sparsity_e, sparsity_i=sparsity_i,
                                                   connection_type=ConnectionType.FIXED_NUMBER)

        print(connections)

        e_connections = connections.copy()
        e_connections[e_connections == -1] = 0
        i_connections = connections.copy()
        i_connections[i_connections == 1] = 0
        i_connections[i_connections == -1] = 1

        print('Numbers e per feature: {}'.format(np.sum(e_connections, axis=0)))
        print('Numbers e per hidden : {}'.format(np.sum(e_connections, axis=1)))
        print(np.sum(e_connections))
        print('Numbers i per feature: {}'.format(np.sum(i_connections, axis=0)))
        print('Numbers i per hidden : {}'.format(np.sum(i_connections, axis=1)))
        print(np.sum(i_connections))

        # Check the shape of the return
        self.assertTrue(np.array_equal(mem.shape[0], connections.shape[1]))
        self.assertTrue(np.array_equal(h_per_mem, connections.shape[0]))

        for row in connections:
            # There should always be exactly 4 ones and 1 -1s per hidden
            self.assertEqual(4, np.where(row == 1)[0].shape[0])
            self.assertEqual(1, np.where(row == -1)[0].shape[0])


if __name__ == '__main__':
    unittest.main()
