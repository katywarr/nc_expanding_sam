import numpy as np
from impl.connections import ConnectionType, get_sparse_connection_matrix


class HiddenNeurons:
    """
    Maintains connection information pertaining to the hidden neurons as they are matured.
    Performs recall calculations for the hidden neurons.
    """
    def __init__(self, f: int, h: int,
                 conn_f_h: float, conn_h_f_e: float, conn_h_f_i: float,
                 f_h_conn_type: ConnectionType, h_f_conn_type: ConnectionType, h_threshold: float):
        """
        Initialise the hidden neuron population
        :param f: Number of features.
        :param h: Number of hidden neurons allocated per memory.
        :param conn_f_h: Feature to hidden connection sparsity in the trained network.
        :param conn_h_f_e: Hidden to feature excitatory connection sparsity in the trained network.
        :param conn_h_f_i: Hidden to feature inhibitory connection sparsity in the trained network.
        :param f_h_conn_type: Hidden neuron pre-synaptic connection type. ``FixedProbability`` or ``FixedNumber``.
        :param: h_f_conn_type: Feature neuron pre-synaptic connection type. ``FixedProbability`` or `FixedNumber``.
        """
        self.f = f
        self.h = h
        self.conn_f_h = conn_f_h
        self.conn_h_f_e = conn_h_f_e
        self.conn_h_f_i = conn_h_f_i
        self.f_h_conn_type = f_h_conn_type
        self.h_f_conn_type = h_f_conn_type
        self.h_threshold = h_threshold

        self.num_memories_stored = 0
        self.f_h_conns = None
        self.h_f_conns = None

    def mature_hidden_for_memories(self, memories: np.ndarray) -> np.ndarray:
        """
        Matures a set of hidden neurons for a set of memories. This includes:
            * Appending the feature to hidden neuron connections for the memories to ``self.f_h_connections``.
            * Appending the hidden to feature neuron connections for the memories to ``self.h_f_connections``.

        :param memories: the new memories.
        :return: The features in the memory that are positively reinforced. This is useful in
            establishing the attractor point for the memory.
        """
        num_memories = memories.shape[0]
        num_hidden = num_memories * self.h

        # Pre-allocate the storage for all the memories
        self.h_f_conns = np.zeros(shape=(num_hidden, self.f))
        self.f_h_conns = np.zeros(shape=(num_hidden, self.f))

        mem_attractor_points = np.zeros(shape=(num_memories, self.f), dtype=int)
        for memory_num in range(num_memories):
            mem_attractor_points[memory_num] = self.__mature_hidden_for_memory(memories[memory_num])
        return mem_attractor_points

    def __mature_hidden_for_memory(self, memory: np.ndarray) -> np.ndarray:
        f_h_conns_this_mem = get_sparse_connection_matrix(memory=memory,
                                                          h=self.h,
                                                          sparsity_e=self.conn_f_h,
                                                          connection_type=self.f_h_conn_type)

        # Hidden to feature neurons for this memory
        polar_memory = memory.copy()
        polar_memory[polar_memory == 0] = -1
        h_f_conns_this_mem = get_sparse_connection_matrix(memory=polar_memory,
                                                          h=self.h,
                                                          sparsity_e=self.conn_h_f_e,
                                                          sparsity_i=self.conn_h_f_i,
                                                          connection_type=self.h_f_conn_type,
                                                          modulation_matrix=True)

        # Add the arrays for this memory to the existing lists
        h_start_index = self.num_memories_stored * self.h
        self.f_h_conns[h_start_index: h_start_index+self.h] = f_h_conns_this_mem
        self.h_f_conns[h_start_index: h_start_index+self.h] = h_f_conns_this_mem
        self.num_memories_stored += 1

        # Return an array length f containing the number of hidden neuron excitatory connections per f. The non-zero
        # values in this array are a subset of the memory.
        conns_e = h_f_conns_this_mem.copy()
        conns_e[conns_e == -1] = 0
        memory_attractor_point = np.sum(conns_e, axis=0, dtype=int)
        memory_attractor_point[memory_attractor_point > 0] = 1

        return memory_attractor_point

    def recall_epoch(self, signal: np.ndarray, ground_truth_label: int):

        epoch_output = np.zeros(shape=(self.f,), dtype=int)

        # Pattern recognition step
        h_neurons_firing, total_f_h_spikes = self.__get_hidden_neurons_firing(signal)

        # Modulation step
        f_neuron_activations = np.zeros(shape=(self.f,))
        for h_neuron_index in range(h_neurons_firing.shape[0]):
            if h_neurons_firing[h_neuron_index]:
                f_neuron_activations += self.h_f_conns[h_neuron_index]
        epoch_output[f_neuron_activations > 0] = 1
        epoch_output[f_neuron_activations <= 0] = 0

        # Collect epoch data
        num_h_firing_per_mem = self.get_h_firing_per_mem(h_neurons_firing=h_neurons_firing)
        total_h_firing = np.sum(num_h_firing_per_mem)
        num_correct_h_fired = num_h_firing_per_mem[ground_truth_label]
        num_incorrect_h_fired = total_h_firing - num_correct_h_fired

        return epoch_output, num_correct_h_fired, num_incorrect_h_fired, total_f_h_spikes

    def get_total_connections(self) -> (int, int, int):
        f_h = int(np.sum(self.f_h_conns))
        h_f_e = int(np.sum(self.h_f_conns[self.h_f_conns == 1]))
        h_f_i = int(-np.sum(self.h_f_conns[self.h_f_conns == -1]))
        return f_h, h_f_e, h_f_i

    def __get_hidden_neurons_firing(self, post_synaptic_input: np.ndarray) -> (np.ndarray, int):
        """
        Returns information on the hidden neurons firing and diagnostic data, based on the post_synaptic input to the
        hidden neuron population.

        :param post_synaptic_input: Post synaptic input to the hidden neuron population
        :return: An array of the hidden neurons that fired and the number of feature to hidden spikes
        """
        activation_function = activation_func(post_synaptic_input)
        h_firing = np.apply_along_axis(activation_function, axis=1, arr=self.f_h_conns)

        total_f_h_spikes = np.sum(h_firing)
        h_firing[h_firing < self.h_threshold] = 0
        h_firing[h_firing >= self.h_threshold] = 1

        return h_firing, total_f_h_spikes

    def get_h_firing_per_mem(self, h_neurons_firing: np.ndarray):
        memory_activations = np.zeros(shape=(self.num_memories_stored,))
        for mem in range(self.num_memories_stored):
            start_index = mem * self.h
            end_index = start_index + self.h
            memory_activations[mem] = np.sum(h_neurons_firing[start_index:end_index])
        return memory_activations


def activation_func(post_synaptic_input: np.ndarray):
    """
    Returns an activation function representative of the most basic artificial neuron: simply a sum of the
    post_synaptic activations.
    :param post_synaptic_input:
    :return:
        lambda defining neuron activation function behaviour.
    """
    return lambda input_array: np.sum(np.logical_and(post_synaptic_input, input_array))
