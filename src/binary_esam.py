import numpy as np
import time
from esam_reporter import ESAMReporter
from impl.hidden_neuron_population import HiddenNeurons
from impl.connections import get_connection_type


class BinaryESAM:
    """
    Binary Expanding Sparse Associative Memory (ESAM) network.
    """
    def __init__(self,
                 network_params: dict):
        """
        Initialise the network based on a dictionary of network parameters.
        :param network_params:
            a dictionary of parameters that describes the ESAM network:
                ``f``: The number of feature neurons in the network.
                ``h``: The number of hidden neurons for each memory.
                ``f_h_sparsity``: Active feature to hidden connection sparsity. An active feature is a feature neuron
                that is activated for the hidden neuron's memory.
                ``h_f_sparsity_e``: Hidden to active feature excitatory connection sparsity. An active feature is a
                feature neuron that is activated for the hidden neuron's memory.
                ``h_f_sparsity_i``: Hidden to inactive feature inhibitory connection sparsity.
                An inactive feature is a feature neuron that is *not* activated for the hidden neuron's memory.
                ``e``: Number of epochs per recall.\n
                ``h_thresh``: The hidden neuron firing threshold (theta).
                ``f_h_conn_type`` and ``h_f_conn_type`` are strings describing the type of network connections
                established from the feature to hidden neurons and hiddden to feature neurons respectively.
                These can have values ``FixedProbability`` or ``FixedNumberPre``. ``FixedNumberPre``
                optimises the connections so there is consistent number of pre-synaptic connections to all the
                neurons in the target population.
        """
        self.initialised = False
        self.num_memories = 0
        self.memories = None

        # Optional parameters
        if 'f_h_conn_type' not in network_params:
            network_params['f_h_conn_type'] = 'FixedProbability'
        if 'h_f_conn_type' not in network_params:
            network_params['h_f_conn_type'] = 'FixedProbability'
        if 'debug' not in network_params:
            network_params['debug'] = False
        # Network Parameter Initialisation
        try:
            self.h = network_params['h']
            self.f = network_params['f']
            self.conn_f_h = network_params['f_h_sparsity']
            self.epochs_per_recall = network_params['e']
            self.h_threshold = network_params['h_thresh']
            self.conn_h_f_e = network_params['h_f_sparsity_e']
            self.conn_h_f_i = network_params['h_f_sparsity_i']
            f_h_conn_type_str = network_params['f_h_conn_type']
            h_f_conn_type_str = network_params['h_f_conn_type']
        except KeyError as e:
            print('Error: The following key was missing from the network parameters (net_params) passed to '
                  'BinaryESAM: ', e)
            raise e
        self.f_h_conn_type = get_connection_type(f_h_conn_type_str)
        self.h_f_conn_type = get_connection_type(h_f_conn_type_str)

        if not self.__validate_initialisation_parameters():
            msg = 'Network parameter validation failed - exiting ESAM initialisation'
            raise ValueError(msg)

        self.reporter = ESAMReporter(network_params=network_params)

        self.f_h_connections = None
        self.h_f_connections = None
        self.num_memories_stored = 0
        self.h_to_mem_mapping = None
        self.hidden_neuron_pop = HiddenNeurons(f=self.f,
                                               h=self.h,
                                               conn_f_h=self.conn_f_h,
                                               conn_h_f_e=self.conn_h_f_e,
                                               conn_h_f_i=self.conn_h_f_i,
                                               f_h_conn_type=self.f_h_conn_type,
                                               h_f_conn_type=self.h_f_conn_type,
                                               h_threshold=self.h_threshold)
        self.initialised = True

    # -----------------------------------------------------------------------------------------------------------------
    # Public methods below
    # -----------------------------------------------------------------------------------------------------------------
    def learn(self, memory_signals: np.ndarray) -> bool:
        """
        Learns a list of memory signals.
        :param memory_signals:
            array of memories, each of length f.
        :return: Flag indicating success.
        """
        memory_attractor_points = np.zeros(shape=memory_signals.shape, dtype=int)   # Will convert to bool on return
        print('>> Learning {} memories'.format(memory_signals.shape[0]))
        if memory_signals.shape[1] != self.f:
            print('Error: Memories passed to signal learner are of incorrect length: {}. They should be length: {}.'
                  .format(memory_signals.shape[1], self.f))
            return False

        real_time_start = time.time()

        memory_attractor_points += self.hidden_neuron_pop.mature_hidden_for_memories(memories=memory_signals)

        real_time_end = time.time()
        total_learning_time = real_time_end - real_time_start
        time_per_memory = round(total_learning_time/memory_signals.shape[0] * 1000, 2)
        print('{} memories learnt in {}s. Approx time per memory: {}ms'
              .format(memory_signals.shape[0], round(total_learning_time, 2), time_per_memory))

        self.reporter.add_memory_signals(memory_signals.astype(int), memory_attractor_points.astype(int))
        self.num_memories_stored += memory_signals.shape[0]

        return True

    def recall(self, test_signals: np.ndarray, ground_truth_labels: np.ndarray) -> ESAMReporter:
        """
        Performs recall on each of the test_signals passed, each of length f, returning an ``ESAMReporter``
        and the average time taken per recall.

        Results of each test (each signal recall) are collated and managed by the returned ``ESAMReporter``.

        The ground truths passed on input enables each result to be compared with what was expected to establish the
        accuracy of the recall.

        :param test_signals:
            array of test signals, each of length f.
        :param ground_truth_labels:
            array of ground truth labels, one for each test signal.
        :return:
            ESAMReporter object.
        """
        if not (self.__validate_signal_data(test_signals, 'test_signals')):
            return self.reporter

        test_signals_recall = test_signals.astype(int)
        print('\n>> Recalling {} tests.'.format(test_signals_recall.shape[0]))

        total_test_time = 0.0
        for simulation_num, signal in zip(range(test_signals_recall.shape[0]), test_signals_recall):
            ground_truth_label = int(ground_truth_labels[simulation_num])
            real_time_start = time.time()
            signal_recall_output, num_correct_h_fired, num_incorrect_h_fired, total_f_h_spikes = \
                self.__recall_signal(signal, ground_truth_label)
            real_time_end = time.time()
            total_test_time = total_test_time + (real_time_end - real_time_start)
            self.reporter.add_simulation_result(recall_output=signal_recall_output,
                                                ground_truth_label=ground_truth_label,
                                                num_f_h_spikes=total_f_h_spikes,
                                                num_correct_h_fired=num_correct_h_fired,
                                                num_incorrect_h_fired=num_incorrect_h_fired)

        time_per_recall = round(total_test_time/test_signals_recall.shape[0] * 1000, 2)
        print('Recall complete. Average time per test (ms) was: ', time_per_recall)
        return self.reporter

    # -----------------------------------------------------------------------------------------------------------------
    # Private methods below
    # -----------------------------------------------------------------------------------------------------------------
    def __recall_signal(self, signal: np.ndarray, ground_truth_label: int) \
            -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Perform a recall based on an occluded signal. Return the resulting signal and diagnostic data.

        The tuple of returned arrays represents -

        * The signal for each epoch, plus the original signal (epoch 0). shape = (no epochs+1, f)
        * The number of correct hidden units firing for each epoch. shape = (epochs,)
        * The number of incorrect hidden units firing for each epoch. shape = (epochs,)
        * The total number of spikes for each epoch. shape = (epochs,)

        :param signal:
            the original signal passed for recall.
        :param ground_truth_label:
            a label identifying the ground truth from which the signal originated.
        :return:
            a tuple of arrays representing the output and diagnostic data.
        """
        recall_output = np.zeros(shape=(self.epochs_per_recall + 1, self.f), dtype=int)

        # First recall value is the original signal
        recall_output[0] = signal.copy()
        num_correct_h_fired = np.full(shape=(self.epochs_per_recall,), fill_value=0, dtype=int)
        num_incorrect_h_fired = np.full(shape=(self.epochs_per_recall,), fill_value=0, dtype=int)
        total_f_h_spikes = np.full(shape=(self.epochs_per_recall,), fill_value=0, dtype=int)
        signal_length = np.full(shape=(self.epochs_per_recall,), fill_value=0, dtype=int)

        for epoch in range(self.epochs_per_recall):
            recall_output[epoch+1], \
                num_correct_h_fired[epoch], \
                num_incorrect_h_fired[epoch], \
                total_f_h_spikes[epoch] = \
                    self.hidden_neuron_pop.recall_epoch(signal=signal, ground_truth_label=ground_truth_label)
            signal_length[epoch] = np.sum(signal)
            signal = recall_output[epoch+1]

        return recall_output, num_correct_h_fired, num_incorrect_h_fired, total_f_h_spikes

    def __validate_signal_data(self, signals: np.ndarray, array_name: str = None) -> bool:
        """
        Validates that a numpy array of signals conforms to that expected.
        :param signals:
            numpy array of signals to be tested.
        :param array_name:
            a readable string identifier to be included in the error message.
        :return:
            True if the data passed is valid, False otherwise.
        """

        if array_name is None:
            array_name = str('signals')
        if signals.shape[1] != self.f:
            print('Error: The {} in the signal array of shape {} do not have the the correct length: {}'
                  .format(array_name, signals.shape, self.f))
            return False
        return True

    def __validate_initialisation_parameters(self):
        """
        Validate the initialisation parameters passed to this object.
        :return:
            True if the parameters pass validation, False otherwise.
        """
        if self.conn_f_h > 1 or self.conn_f_h < 0:
            print('ESAM Init Error: The connection probability between feature neurons and hidden neurons must be '
                  'between 0 and 1. \nThe value: {} is invalid.'.format(self.conn_f_h))
            return False

        if self.conn_h_f_e > 1 or self.conn_h_f_e < 0:
            print('ESAM Init Error: The excitatory connection probability between hidden neurons and feature neurons '
                  'must be between 0 and 1. \nThe value: {} is invalid.'.format(self.conn_h_f_e))
            return False

        if self.conn_h_f_i > 1 or self.conn_h_f_i < 0:
            print('ESAM Init Error: The inhibitory connection probability between hidden neurons and feature neurons '
                  'must be between 0 and 1. \nThe value: {} is invalid.'.format(self.conn_h_f_i))
            return False

        if self.f < 1:
            print('ESAM Init Error: The size of each signal (number of features) must be greater than one. '
                  'The value: {} is invalid.'.format(self.f))
            return False

        return True

