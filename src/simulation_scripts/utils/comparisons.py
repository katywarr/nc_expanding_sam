import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# Comparison functions
# Simple comparison functions that take two 1 dimensional arrays of equal length and return a comparison score (integer
# or float).
# ---------------------------------------------------------------------------------------------------------------------

def sum_differences(signal1, signal2):
    """
    Returns the number of values that differ between two equal length binary signals
    :param signal1:
    :param signal2:
    :return:
    """
    return np.sum(np.logical_xor(signal1, signal2))

def proportion_similarity(signal1: np.ndarray, signal2: np.ndarray, round_amount: int = 4) -> float:
    return round(np.sum(signal1 == signal2)/len(signal1), round_amount)

def percent_similarity(signal1: np.ndarray, signal2: np.ndarray, round_amount: int = 4) -> float:
    return round(proportion_similarity(signal1, signal2, round_amount+2) * 100, round_amount)

def proportion_similarity_binary(signal1: np.ndarray, signal2: np.ndarray, round_amount: int = 4) -> float:
    length = signal1.shape[0]
    differences = np.sum(np.logical_xor(signal1, signal2))
    return round((length-differences)/length, round_amount)

def percent_similarity_binary(signal1: np.ndarray, signal2: np.ndarray, round_amount: int = 4) -> float:
    return round(proportion_similarity_binary(signal1, signal2, round_amount+2) * 100, round_amount)

# ---------------------------------------------------------------------------------------------------------------------
# Higher level comparison functions
# ---------------------------------------------------------------------------------------------------------------------
def compare_tests_to_memories(test_signals: np.ndarray, memories: np.ndarray, comparison_function):
    """
    For each binary test signal, apply a comparison function against each memory that returns a single value to
    establish a similarity score.
    Return all the results in an array.

    :param test_signals:
    :param memories:
    :param comparison_function = None
    A comparison function that takes two 1d array arguments. The second argument name must be 'signal2'
    If no comparison function is provided, the comparison defaults to the sum_differences function in this file.
    :return:
    Return all the results in an array of shape=(test_signals.shape[0], memories.shape[0])
    """

    if (test_signals.shape[1] != memories.shape[1]) or len(test_signals.shape) != 2 or len(memories.shape) != 2:
        print('Error: compare_tests_to_memories was passed invalid numpy arrays.\n'
              '       test_signals: {}   memories: {}'
              .format(test_signals.shape, memories.shape))
        return None
    comparisons = np.full(shape=(test_signals.shape[0], memories.shape[0]), fill_value=-1.0)
    for i, test_signal in zip(range(test_signals.shape[0]), test_signals):
        comparisons[i] = np.apply_along_axis(func1d=comparison_function,
                                             axis=1,
                                             arr=memories,
                                             signal2=test_signal)
    return comparisons


def compare_binary_signals(signal1, signal2):
    length = signal1.shape[0]
    differences = np.sum(np.logical_xor(signal1, signal2))
    percent_similarity = round((length-differences)/length*100, 1)
    return percent_similarity, differences
