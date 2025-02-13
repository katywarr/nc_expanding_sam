import unittest
from data_generation.data_generator import generate_signals
import numpy as np

class DataGeneratorTest(unittest.TestCase):

    def test_generate_signals(self):
        active = 3
        num_signals = 10
        f = 20
        signals = generate_signals(num_signals=num_signals,
                                   f=f,
                                   active_features=active)

        self.assertEqual(signals.shape[0], num_signals)  # add assertion here
        self.assertEqual(signals.shape[1], f)
        self.assertEqual(np.count_nonzero(signals), active * num_signals)

        count_per_signal = np.count_nonzero(signals, axis=1)
        self.assertEqual(len(count_per_signal[count_per_signal != active]), 0)

        count_per_feature = np.count_nonzero(signals, axis=0)
        self.assertGreater(np.sum(count_per_feature[active:-1]), 0) # Check some shuffling


if __name__ == '__main__':
    unittest.main()
