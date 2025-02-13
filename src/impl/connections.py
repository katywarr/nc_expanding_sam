from enum import IntEnum
import numpy as np


class ConnectionType(IntEnum):
    FIXED_PROBABILITY = 1
    FIXED_NUMBER = 2

def get_connection_type(type_as_str: str) -> ConnectionType:
    match type_as_str:
        case 'FixedProbability':
            return ConnectionType.FIXED_PROBABILITY
        case 'FixedNumberPre':
            return ConnectionType.FIXED_NUMBER

    raise ValueError('Connection type {} passed in network definition is not recognised.'.format(type_as_str))


def get_sparse_connection_matrix(memory: np.ndarray,
                                 h: int,
                                 sparsity_e: float,
                                 sparsity_i: float = 0,
                                 connection_type: ConnectionType = ConnectionType.FIXED_PROBABILITY,
                                 modulation_matrix: bool = False):
    """
    Return an array the same memory repeated (one for each hidden neuron) that has been masked (i.e. values set to
    zero) according to the connection probabilities passed on input. The returned connections array is
    of shape (f,h) where f is the length of the memory. Excitatory connections are 1 and inhibitory
    connections are -1.

    This function can be used to generate both the feature to hidden neuron sparse connections and the hidden to
    feature neuron sparse connections.

    :param memory:
        The memory as a binary array or as a binary polar (1 and -1s) array if inhibitory connections.
    :param h:
        The number of hidden neurons per memory. This defines the number of repeated elements that will be in the
        returned connection array.
    :param sparsity_e:
        The excitatory connection sparsity where sparsity is either a probability or proportion
        (see sparsity_is_probability flag).
        1-conn_prob_e of the 1s in the repeated memory array will be set to zero - i.e. no connection.
    :param sparsity_i:
        The inhibitory connection sparsity where sparsity is either a probability or proportion
        (see sparsity_is_probability flag).
        1-conn_prob_i of the -1s in the repeated memory array will be set to
        zero - i.e. no connection.
    :param connection_type:
        Connection sparsity can be treated as a probability or a proportion. Set this to False in order to treat it as
        a probability and True to treat as a proportion.
        Default is False.
    :param modulation_matrix:
        By default, this parameter is set to False and the matrix returned will be for the pattern recognition step
        where the connections are feature to hidden.
        By setting this to true, the matrix returned is for the modulation step where the connections are hidden to
        feature.
        The setting of this parameter only affects the interpretation of connection_type. In the feature to hidden
        (modulation_matrix == False) case, the feature neurons are the pre-synaptic ones and hidden are post synaptic.
        Opposite applies when this parameter is set to True.
    :return:
    """
    f = memory.shape[0]
    if connection_type == ConnectionType.FIXED_PROBABILITY:
        # Repeat the memory for each of the hidden neurons
        connections = np.repeat([memory.astype(int)], repeats=h, axis=0)
        # Randomly remove values according to the connection probability
        random_array = np.random.rand(h, f)
        connections[(random_array > sparsity_e) & (connections == 1)] = 0
        connections[(random_array > sparsity_i) & (connections == -1)] = 0
    else:
        connections = np.zeros(shape=(h, f), dtype=int)

        f_idx_e = np.where(memory == 1)[0]
        if sparsity_i == 0:
            f_idx_i = np.array([])                  # Purely for performance if inhibitory not used
        else:
            f_idx_i = np.where(memory == -1)[0]

        if (connection_type == ConnectionType.FIXED_NUMBER) & (not modulation_matrix):
            # Same number of connections per hidden neuron
            # Calculate the number of connections to add
            # Note that rounding to int means that the number could be slightly more/less than the average.
            num_e_to_add_per_h = round(sparsity_e * f_idx_e.shape[0])
            num_i_to_add_per_h = round(sparsity_i * f_idx_i.shape[0])

            # Loop through each of the hidden neurons and randomly add the connections
            for hidden in range(h):
                h_instance = connections[hidden]

                np.random.shuffle(f_idx_e)
                locs_e_to_add = f_idx_e[0:num_e_to_add_per_h].copy()
                for loc_e in locs_e_to_add:
                    h_instance[loc_e] = 1

                np.random.shuffle(f_idx_i)
                locs_i_to_add = f_idx_i[0:num_i_to_add_per_h].copy()
                for loc_i in locs_i_to_add:
                    h_instance[loc_i] = -1
        else:
            h_idxs = np.arange(0, h)
            if sparsity_e != 0:
                # Calculate the number of excitatory connections to add per feature neuron
                num_e_to_add_per_f = round(sparsity_e * h)
                for e_feature in f_idx_e:
                    np.random.shuffle(h_idxs)
                    h_to_add = h_idxs[0:num_e_to_add_per_f]
                    connections[h_to_add, e_feature] = 1

            if sparsity_i != 0:
                # Calculate the number of inhibitory connections to add per feature neuron
                num_i_to_add_per_f = round(sparsity_i * h)
                for i_feature in f_idx_i:
                    np.random.shuffle(h_idxs)
                    h_to_add = h_idxs[0:num_i_to_add_per_f]
                    connections[h_to_add, i_feature] = -1

    return connections
