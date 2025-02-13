from scipy.stats import binom

def n_choose_k_with_prob(n, k, prob_k):
    """
    Returns the probability that, given k choices, each with prob_k of success, from n independent binomial
    possibilities, that all the choices will be successful.
    :param n: The total number to select from.
    :param k: The number of selections to make.
    :param prob_k: The probability that a single selection from n is successful.
    :return: The confidence (probability) that a selection k is successful.
    """
    # prob = math.comb(n, k) * (prob_k ** k) * ((1 - prob_k) ** (n - k))
    # Using the probability mass function provided by is more efficient.
    prob = binom.pmf(k, n, prob_k)
    return prob


def n_choose_fewer_than_k_with_prob(n, k, prob):
    """
    Sums the probabilities that, given 0 to k-1 choices, each with prob of success, from n independent binomial
    possibilities, that all the choices will be successful.
    :param n:
        The total number to select from.
    :param k:
        The number of selections to make.
    :param prob:
        The probability that a single selection from n is successful.
    :return:
        The confidence (probability) that a selection k is successful.
    """
    return_prob = 0
    for select in range(0, k):
        return_prob += n_choose_k_with_prob(n, select, prob)
    return return_prob


def prob_f_h_carries_signal(recall_type:str, s_m: float, s_n: float):
    """
    Given a learnt connection, ``prob_f_h_carries_signal`` describes the probability that the connection
    will carry a signal during recall. This varies depending on the recall type.

    * **Equation 6.5** in the paper when the recall type is `correct`
    * **Equation 6.6** in the paper when the recall type is `perfect` (correct and no noise)
    * **Equation 6.7** in the paper when the recall type is `incorrect`

    :param recall_type:
        The nature of the recall probability that is being calculated: `correct`, `incorrect`, or `perfect`.
    :param s_m:
        Memory sparsity.
    :param s_n:
        Sparsity of the noise added to each recall signal.
    :return:
    """
    if recall_type == 'correct':
        # With a noisy memory, the probability that an f-h learned connection is activated is the prob of a
        # learned connection is 1 multiplied by (1 minus the probability that the learned signal contains noise)
        # which equates to 1-noise.
        prob = 1 - s_n
    else:
        if recall_type == 'incorrect':
            # With a memory unknown to the hu, the probability that an f-h learned connection is activated is the
            # probability that that bit of the recall signal is a 1.
            s_r = s_m * (1-s_n) + ((1-s_m) * s_n)      # Recall signal sparsity
            prob = s_r
        else:
            if recall_type == 'perfect_mem':
                # Shorthand for 'correct' when s_n == 0
                prob = 1
            else:
                raise ValueError('Recall type: '+recall_type+' is not valid.')

    # print('Probability of signal on a connection for recall type {} is: {}\n'.format(recall_type, prob))
    return prob


def prob_conns(num_conns: int,
               pre_population_size: int,
               conn_sparsity: float,
               conn_type: str):
    """
    Returns the probability that a specific number of pre-synaptic connections (num_conns) will be established to
    a hidden neuron.

    **Equation 6.2** in the paper.

    :param num_conns:
        Number of connections that the return probability refers to (``c`` i in equation 6.2).
    :param pre_population_size:
        Number of pre-synaptic connections (``s_m * f`` in equation 6.2).
    :param conn_sparsity:
        The sparsity of connections from the pre-synaptic sub-set (``s+_{f_h}`` in equation 6.2).
    :param conn_type:
        The connection rule used in establishing the connections:
        ``FixedProbability`` or ``FixedNumber``
    """
    prob = 0
    if conn_type == 'FixedProbability':
        prob = n_choose_k_with_prob(pre_population_size, num_conns, conn_sparsity)
    else:
        if conn_type == 'FixedNumberPre':
            static_c = round(pre_population_size * conn_sparsity)
            if num_conns == static_c:
                prob = 1
        else:
            raise ValueError('Connection type: ' + conn_type + ' is not valid.')
    # print('Probability of {} connections is: {}'.format(num_conns, prob))
    return prob


def prob_activation(a: int,
                    recall_type: str,
                    max_connections: int,
                    signal_fn,
                    connection_fn):
    """
    Returns the probability that a hidden neuron will reach exactly activation `a`.
    *Equation 6.3 in the paper.*

    :param a: activation level being tested.
    :param recall_type: Type of recall for which the probability is being calculated. `correct`, `incorrect`,
        or `perfect_mem`.
    :param max_connections: Maximum pre-synaptic connections to a hidden neuron.
    :param signal_fn: A function that returns the probability of a signal on a pre-synaptic connection. This is based
        on the sparsity of the memories and the amount of noise added to the recall signals. Refer to
        ``prob_f_h_carries_signal`` for the implementation of the signal function.
    :param connection_fn: A function that returns the probability of a connection.
    :return:
    """
    prob = 0
    prob_signal = signal_fn(recall_type=recall_type)
    for num_conns in range(a, max_connections):
        prob_a_for_num_conns = n_choose_k_with_prob(num_conns, a, prob_signal)
        prob += connection_fn(num_conns) * prob_a_for_num_conns
    return prob


def prob_firing(recall_type: str, theta: int, max_conns: int, activation_fn) -> float:
    """
    Returns the probability of a hidden neuron firing.
    *Equation 6.1 in the paper.*
    :param recall_type:

    :param theta:
        The hidden neuron threshold.
    :param max_conns:
    :param activation_fn:
    :return: probability (float)
    """
    prob = 0
    for activation in range(theta, max_conns):
        prob += activation_fn(activation, recall_type)
    return prob


def prob_correct_pattern_retrieval(m: int,
                                   h: int,
                                   prob_h_firing_correct: float,
                                   prob_h_firing_incorrect: float,
                                   inhibition: bool) -> float:
    """
    Returns the probability of a correct pattern retrieval in a single epoch. The h_firing probabilities passed to
    this function depend on the threshold. The optimum threshold can therefore be established by finding the
    best probability of correct pattern retrieval based on different threshold values.

    Hidden to feature neuron inhibition is assumed to be either 1 or 0 for this function.

    **Equation 6.11** in the paper.

    :param m: Number of memories.
    :param h: Number of hidden neurons per memory.
    :param prob_h_firing_correct: Probability of a hidden neuron firing correctly.
    :param prob_h_firing_incorrect: Probability of a hidden neuron firing incorrectly.
    :param inhibition: If True `h_f_sparsity` is assumed to be 1. Else `h_f_sparsity` is assumed to be 0.
    :return: Probability of correct pattern retrieval (float).
    """

    prob = 0
    incorrect_h = (m - 1) * h
    if inhibition:
        # Equation 6.11 in the paper (equation 3.6 in Hoffmann 2019)
        for num_h_firing in range(1, int(h)+1):

            prob_correct_h = n_choose_k_with_prob(h, num_h_firing, prob_h_firing_correct)
            prob_incorrect_fewer_h = n_choose_fewer_than_k_with_prob(incorrect_h,
                                                                     num_h_firing,
                                                                     prob_h_firing_incorrect)
            prob += prob_correct_h * prob_incorrect_fewer_h
    else:
        # No equivalent equation in the paper for this code, but included here for completeness.
        # (Equation 3.8 in Hoffmann 2019)
        prob = (1 - (1 - prob_h_firing_correct) ** h) * (1 - prob_h_firing_incorrect) ** incorrect_h
    return prob


# ----------------------------------------------------------------------------------------------------------------------
# Lambdas for re-use.
# The following functions return lambdas that can be re-used for a specific scenario.
# ----------------------------------------------------------------------------------------------------------------------
def prob_f_h_carries_signal_fn(s_m: float, s_n: float):
    """
    Return a prob_f_h_carries_signal lambda that can be re-used for the problem space across different recall types
    to determine the probability of a signal on a feature to hidden neuron connection.
    """
    return lambda recall_type: prob_f_h_carries_signal(recall_type=recall_type, s_m=s_m, s_n=s_n)


def prob_conns_fn(length, sparsity, conn_type):
    """
    Return a prob_conns function that can be re-used for a problem space and network, with just the
    number of connections changing.
    """
    return lambda c: prob_conns(num_conns=c,
                                pre_population_size=length,
                                conn_sparsity=sparsity, conn_type=conn_type)


def prob_activation_fn(max_connections, signal_fn, connection_fn):
    """
    Return a prob_activation function that can be re-used for a problem scenario and network with just the activation
    changing.

    :param max_connections: Maximum number of pre-synaptic connections to a hidden neuron.
    :param signal_fn: A function that generates the probability of each connection carrying a spike.
    :param connection_fn: A function that generates the probability of a connection.
    """
    return lambda a, recall_type: prob_activation(a=a,
                                                  recall_type=recall_type,
                                                  max_connections=max_connections,
                                                  signal_fn=signal_fn,
                                                  connection_fn=connection_fn)

def prob_firing_fn(max_conns, activation_fn):
    """
    Return a prob_firing function that can be re-used for a problem scenario with just the recall type and
    value for theta changing.

    :param max_conns: Maximum number of pre-synaptic connections to a hidden neuron.
    :param activation_fn: Açtivation function.

    """
    return lambda recall_type, theta: prob_firing(recall_type=recall_type,
                                                  theta=theta,
                                                  max_conns=max_conns,
                                                  activation_fn=activation_fn)