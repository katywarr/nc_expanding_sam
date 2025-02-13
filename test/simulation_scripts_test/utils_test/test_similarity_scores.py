import unittest
import numpy as np
from simulation_scripts.utils.comparisons import proportion_similarity, percent_similarity, \
    proportion_similarity_binary, percent_similarity_binary, \
    compare_tests_to_memories, sum_differences


class TestSimilarityScores(unittest.TestCase):

    def test_proportion_same_sets(self):
        signal1 = np.array([1, 1, 1, 2, 3, 4])
        signal2 = np.array([2, 3, 4, 1, 1, 1])
        score = proportion_similarity(signal1, signal2)
        self.assertEqual(round(0/6, 4), score)

    def test_proportion_similar_arrays(self):
        signal1 = np.array([1, 1, 1, 2, 3, 4])
        signal2 = np.array([1, 1, 1, 2, 7, 8])
        score = proportion_similarity(signal1, signal2)
        self.assertEqual(round(4/6, 4), score)

    def test_percent_similar_arrays(self):
        signal1 = np.array([1, 1, 1, 2, 3, 4])
        signal2 = np.array([1, 1, 1, 2, 7, 8])
        score = percent_similarity(signal1, signal2, 2)
        self.assertEqual(round(4/6*100, 2), score)

    def test_proportion_binary(self):
        signal1 = np.array([1, 1, 1, 0, 0, 0])
        signal2 = np.array([1, 1, 1, 1, 1, 1])
        score = proportion_similarity_binary(signal1, signal2)
        self.assertEqual(0.5000, score)

    def test_percent_binary(self):
        signal1 = np.array([1, 1, 1, 0, 0, 0])
        signal2 = np.array([1, 0, 1, 1, 1, 1])
        score = percent_similarity_binary(signal1, signal2, 2)
        self.assertEqual(round(2/6*100, 2), score)

    def test_compare_tests_to_memories(self):
        mems_test = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
                             )
        signals_test = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
                                )
        output = compare_tests_to_memories(signals_test, mems_test, sum_differences)
        expected = np.array([[5, 5, 0, 10],
                             [5, 5, 10, 0],
                             [4, 6, 1, 9]])
        self.assertTrue(np.array_equal(expected, output))


if __name__ == '__main__':
    unittest.main()
