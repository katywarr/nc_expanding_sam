from binary_esam import BinaryESAM
from data_generation.data_generator import generate_simulation_data
from esam_reporter import ESAMReporter
from simulation_scripts.utils.threshold_generator import HiddenNeuronThresholdGenerator


def run_simulations(num_simulations: int,
                    problem_space_params: dict,
                    network_params: dict,
                    simulation_data: dict = None) \
        -> ESAMReporter:
    """
    Run a set of recall simulations based on a defined problem space, ESAM network, and simulator. Return an
    ``ESAMReporter`` and the average time per simulation in ms.

    :param num_simulations:
    :param problem_space_params:
        Dictionary of parameters representing the problem space.
    :param network_params:
        Dictionary of parameters representing the network definition.
    :param simulation_data:
        Optional simulation data comprising 'memory_signals', 'test_signals', and 'ground_truth_labels'.
        If this isn't passed, the data will be generated.
        This option is provided to enable exactly the same data to be used across multiple calls to this
        function.
    :return: ESAMReporter and average time per simulation
    """

    reporter = None

    if simulation_data is None:
        memory_signals, test_signals, ground_truth_labels = generate_simulation_data(problem_space=problem_space_params,
                                                                                     num_simulations=num_simulations)
    else:
        memory_signals = simulation_data['memory_signals']
        test_signals = simulation_data['test_signals']
        ground_truth_labels = simulation_data['ground_truth_labels']

    network_params_for_test = network_params.copy()
    if network_params['h_thresh'] == -1:
        network_params_for_test['h_thresh'] = (
            generate_hidden_neuron_threshold(problem_space=problem_space_params,
                                             net_param_h=network_params['h'],
                                             net_param_f_h_sparsity=network_params['f_h_sparsity'],
                                             net_param_f_h_conn_type=network_params['f_h_conn_type']))

        print('Using generated threshold: {} \n'.format(network_params_for_test['h_thresh']))

    print('  Learning phase...')
    esam = BinaryESAM(network_params=network_params_for_test)

    if not esam.initialised:
        return None

    if esam.learn(memory_signals=memory_signals):
        print('  Recall phase...')
        reporter = esam.recall(test_signals=test_signals, ground_truth_labels=ground_truth_labels)

    print('Simulation complete')

    return reporter


def generate_hidden_neuron_threshold(problem_space: dict,
                                     net_param_h: int,
                                     net_param_f_h_sparsity: float,
                                     net_param_f_h_conn_type: str,
                                     ):
    """
    Given the network and the problem space described by the input parameters, return the optimum threshold.
    Because this method is expensive for many tests, we cache the probability results used to calculate the
    optimum threshold in a file (this occurs in the ``HiddenNeuronParameters`` object).
    :param problem_space:
        Dictionary of parameters describing the problem space.
    :param net_param_h:
        Hidden neurons per memory.
    :param net_param_f_h_sparsity:
        Sparsity of the feature to hidden neuron connections.
    :param net_param_f_h_conn_type:
        Connection type as string FixedProbability of FixedNumberPre.
    :return:
    """

    print('Establishing hidden neuron threshold for the following test:'
          '\n    net_param_f:             {}'
          '\n    net_param_h:             {}'
          '\n    net_param_f_h_sparsity:  {}'
          '\n    net_param_f_h_conn_type: {}'
          '\n    data_param_s_m:          {}'
          '\n    data_param_s_n:          {}'
          '\n    data_param_m:            {}'
          .format(problem_space['f'], net_param_h, net_param_f_h_sparsity, net_param_f_h_conn_type, problem_space['s_m'],
                  problem_space['s_n'], problem_space['m']))

    hp_generator = HiddenNeuronThresholdGenerator(problem_space=problem_space,
                                                  net_param_h=net_param_h,
                                                  net_param_f_h_sparsity=net_param_f_h_sparsity,
                                                  net_param_f_h_conn_type=net_param_f_h_conn_type,
                                                  )

    recommended_threshold, recommended_net_param_h = hp_generator.get_best_threshold(m=problem_space['m'])

    av_conns_per_h = int(problem_space['f'] * problem_space['s_m'] * net_param_f_h_sparsity)

    print('    The recommended hidden neuron threshold is {}'.format(recommended_threshold))
    if net_param_f_h_conn_type != 'FixedProbability':
        print('    Using static f-h connections proportion: {} gives exactly {} connections per hidden neuron.'
              .format(net_param_f_h_sparsity, av_conns_per_h))
        if av_conns_per_h < recommended_threshold:
            required_sparsity = recommended_threshold / (problem_space['f'] * problem_space['s_m'])
            print('    WARNING: The hidden neurons do not have sufficient connections to fire.\n'
                  '             Consider increasing the f_h sparsity to at least {}.'.format(required_sparsity))
    else:
        print('    Using probability of f-h connections per hidden neuron: {} gives an average of : {} connections\n'
              .format(net_param_f_h_sparsity, av_conns_per_h))
        if recommended_net_param_h > net_param_h:
            print('    Warning: {} hidden neurons have been allocated per memory, but some will have insufficient '
                  'connections to ever fire (they will be redundant).\n'
                  '             To ensure {} non-redundant hidden neurons, '
                  'consider increasing the f_h sparsity or increase the number of hidden neurons for allocated'
                  'to each memory to {}.'.format(net_param_h, net_param_h, recommended_net_param_h))

    return recommended_threshold
