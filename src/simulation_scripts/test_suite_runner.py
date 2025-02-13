import os
import numpy as np
from simulation_scripts.utils.excel_handler import ExcelHandler, get_timestamp_str, check_output_dir
from data_generation.data_generator import generate_simulation_data
from simulation_scripts.simulation_runner import run_simulations
import pandas as pd
import collections.abc


class TestSuiteRunner:

    def __init__(self,
                 problem_space_default: dict,
                 network_default: dict,
                 output_dir: str,
                 seed: int = 5):

        check_output_dir(output_dir=output_dir)
        self.output_dir = output_dir
        self.problem_space_default = problem_space_default
        self.network_default = network_default
        self.seed = seed

    def run_vary_problem_space_test_suite(self,
                                          all_tests: dict,
                                          num_tests: int,
                                          sims_per_test: int,
                                          seed: int = None):
        """
        Run a set of tests where, for each test, the network is static and the problem space varies along one or more
        hyperparameter dimensions.
        """

        if seed is not None:
            self.seed = seed

        # Check for mandatory dictionary values in order to fail fast.
        for test_name, test_details in all_tests.items():
            try:
                network_params_static = test_details['network_static']
                problem_space_vary = test_details['problem_vary']
            except KeyError as e:
                print('Error: Test {} was missing a dictionary entry for either \'network_static\' or \'problem_vary\''
                      .format(test_name))
                raise e

        for test_name, test_details in all_tests.items():
            # Retrieve the selection of networks to try for this test
            network_params_static = test_details['network_static']
            problem_space_vary = test_details['problem_vary']

            # Run the tests
            print('\n================================================================================================')
            print('Running test set: {} defined as: '
                  '\n    Static network params:    {}'
                  '\n    Varying problem space:    {}'
                  .format(test_name, network_params_static, problem_space_vary))
            print('\nNumber of tests per scenario: {} \nNumber of simulations per test: {} '
                  .format(num_tests, sims_per_test))

            test_suite_out = self.run_vary_problem_space_test(network_params=network_params_static,
                                                              problem_space_test_variations=problem_space_vary,
                                                              num_tests_per_test_spec=num_tests,
                                                              num_sims_per_test=sims_per_test)

            self.write_test_results_to_file(test_suite_type='vary_problem', static_params=network_params_static,
                                            test_name=test_name, test_suite_out=test_suite_out, all_tests=all_tests)

            print('Running test set: {} COMPLETE'.format(test_name))

    def run_vary_problem_space_test(self,
                                    network_params: dict,
                                    problem_space_test_variations: dict,
                                    num_tests_per_test_spec: int,
                                    num_sims_per_test: int) -> (pd.DataFrame, pd.DataFrame):
        """
        Run a set of tests where the network remains the same and the problem space is varied.

        :param network_params:
            Static network definition for the set of tests
        :param problem_space_test_variations:
            Problem space parameters. A list is provided for parameters that need to be varied.
        :param num_tests_per_test_spec:
            Number of tests (networks to instantiate) for each network specification.
        :param num_sims_per_test:
            Number of recalls per test. The total number of recalls will be num_tests_per_test_spec * num_sims_per_test
        :return:
            Pandas dataframe containing the results of each simulation.
        """
        problem_varying, num_test_specifications = expand_params_for_tests(problem_space_test_variations)
        sim_results_all = pd.DataFrame()

        for test_spec in range(num_test_specifications):
            # Reset the seed for consistency for this test specification
            np.random.seed(self.seed)

            problem_space = {'f': problem_varying['f'][test_spec],
                             'm': problem_varying['m'][test_spec],
                             's_m': problem_varying['s_m'][test_spec],
                             's_n': problem_varying['s_n'][test_spec]}
            network_params['f'] = problem_space['f']

            # For each scenario run num_tests_per_test_spec with the relevant number of simulations per test
            # (captured in the run_simulations function).
            # Because the problem space is varying, we can't use the same data for all the tests (unlike the
            # network_vary test suite where the problem space remains static).
            for simulation in range(num_tests_per_test_spec):
                print('\n--------------------------------------------------------------------------------------------')
                print('Simulation {} of {} for test {} of {}\n'.format(simulation+1, num_tests_per_test_spec,
                                                                     test_spec+1, num_test_specifications))
                reporter = run_simulations(num_sims_per_test,
                                              network_params=network_params,
                                              problem_space_params=problem_space,
                                              simulation_data=None)  # Generate the data
                if reporter is not None:
                    reporter.set_ground_truth_interpretation(False)
                    all_sims_for_test = reporter.get_test_data_averages(problem_space=problem_space)
                    sim_results_all = pd.concat([sim_results_all, all_sims_for_test])

        return sim_results_all

    def run_vary_network_test_suite(self,
                                    all_tests: dict,
                                    num_tests: int,
                                    sims_per_test: int,
                                    seed: int = 5):
        """
        Run a set of tests where, for each test, the problem space is static and the network varies along one or more
        hyperparameter dimensions.

        :param all_tests:
        :param num_tests:
        :param sims_per_test:
        :param seed:
        :return:
        """
        check_output_dir(output_dir=self.output_dir)

        if seed is not None:
            self.seed = seed

        # Check for mandatory dictionary values in order to fail fast.
        for test_name, test_details in all_tests.items():
            try:
                problem_space_static = test_details['problem_static']
                network_params_vary = test_details['network_vary']
            except KeyError as e:
                print('Error: Test {} was missing a dictionary entry for either \'problem_static\' or \'network_vary\''
                      .format(test_name))
                raise e

        for test_name, test_details in all_tests.items():
            # Retrieve the selection of networks to try for this test
            problem_space_static = test_details['problem_static']
            network_params_vary = test_details['network_vary']
            # Run the tests
            print('\n================================================================================================')
            print('Running test set: {} defined as: '
                  '\n    Static problem space:    {}'
                  '\n    Varying network params:  {}'
                  .format(test_name, problem_space_static, network_params_vary))
            print('\nNumber of tests per scenario: {} \nNumber of simulations per test: {} '
                  .format(num_tests, sims_per_test))
            test_suite_out = self.run_vary_network_test(problem_space=problem_space_static,
                                                        network_params_test_variations=network_params_vary,
                                                        num_tests_per_test_spec=num_tests,
                                                        num_sims_per_test=sims_per_test)

            self.write_test_results_to_file(test_suite_type='vary_network', static_params=problem_space_static,
                                            test_name=test_name, test_suite_out=test_suite_out, all_tests=all_tests)

            print('Running test set: {} COMPLETE'.format(test_name))

    def run_vary_network_test(self,
                              problem_space: dict,
                              network_params_test_variations: dict,
                              num_tests_per_test_spec: int,
                              num_sims_per_test: int) -> pd.DataFrame:
        """
        Run a set of tests where the problem space remains the same and the network architecture is varied.

        :param problem_space:
            Static problem space for the set of tests
        :param network_params_test_variations:
            Network parameters. A list is provided for parameters that need to be varied.
        :param num_tests_per_test_spec:
            Number of tests (networks to instantiate) for each network specification.
        :param num_sims_per_test:
            Number of recalls per test. The total number of recalls will be num_tests_per_test_spec * num_sims_per_test
        :return:
            Pandas dataframe containing the results of each simulation.
        """

        network_varying, num_test_specifications = expand_network_params_for_tests(network_params_test_variations)

        sim_results_all = pd.DataFrame()

        sim_data_list = []
        for tests in range(num_tests_per_test_spec):
            # The same data is used for each test.
            sim_data = {}
            sim_data['memory_signals'], sim_data['test_signals'], sim_data['ground_truth_labels'] = \
                generate_simulation_data(problem_space=problem_space, num_simulations=num_sims_per_test)
            sim_data_list.append(sim_data)

        for test_spec in range(num_test_specifications):
            # Reset the seed for consistency for this test specification
            np.random.seed(self.seed)

            network_params = {'f': network_varying['f'][test_spec],
                              'h': network_varying['h'][test_spec],
                              'f_h_sparsity': network_varying['f_h_sparsity'][test_spec],
                              'h_f_sparsity_e': network_varying['h_f_sparsity_e'][test_spec],
                              'h_f_sparsity_i': network_varying['h_f_sparsity_i'][test_spec],
                              'e': network_varying['e'][test_spec],
                              'h_thresh': network_varying['h_thresh'][test_spec],
                              'f_h_conn_type': network_varying['f_h_conn_type'][test_spec],
                              'h_f_conn_type': network_varying['h_f_conn_type'][test_spec],
                              }
            problem_space['f'] = network_params['f']

            for opt_param, opt_param_type in zip(['cols', 'f_h_conn_type', 'h_f_conn_type'], [int, str, str]):
                if opt_param in network_varying:
                    network_params[opt_param] = network_varying[opt_param][test_spec].astype(opt_param_type)

            # For each scenario run num_tests_per_test_spec with the relevant number of simulations per test (captured
            # in the run_simulations function).
            # Note that, because the problem space remains for all the test specifications, we can use the previously
            # generated sim_data for consistency of testing across the test specifications.
            for simulation, sim_data in zip(range(num_tests_per_test_spec), sim_data_list):
                print('\n--------------------------------------------------------------------------------------------')
                print('Simulation {} of {} for test {} of {}'.format(simulation+1, num_tests_per_test_spec,
                                                                     test_spec+1, num_test_specifications))
                reporter = run_simulations(num_sims_per_test,
                                           network_params=network_params,
                                           problem_space_params=problem_space,
                                           simulation_data=sim_data)

                if reporter is not None:
                    reporter.set_ground_truth_interpretation(False)
                    all_sims_for_test = reporter.get_test_data_averages(problem_space=problem_space)
                    sim_results_all = pd.concat([sim_results_all, all_sims_for_test])

        return sim_results_all

    def write_test_results_to_file(self, test_suite_type: str, static_params: dict,
                                   test_name: str,
                                   all_tests: dict,
                                   test_suite_out: pd.DataFrame):

        if test_suite_out.empty:
            raise KeyError('Unable to write test results to fila as no test results found for test name: '+test_name)

        # Save the data
        save_file_full_name = self.output_dir + os.sep + test_name
        test_summary_writer = ExcelHandler(self.output_dir, test_name)
        print('Writing full simulation data for tests to file: {}'.format(save_file_full_name + '.xlsx'))
        print('... Writing test description sheets (will override previous)')

        # Data that is consistent across all the test variations
        if test_suite_type == 'vary_network':
            test_summary_writer.add_sheet(df=pd.DataFrame(static_params, index=[0]),
                                          sheet_name='problem_space_static')
            test_summary_writer.add_sheet(df=pd.DataFrame(self.network_default, index=[0]),
                                          sheet_name='network_default')
        else: # vary_problem
            test_summary_writer.add_sheet(df=pd.DataFrame(static_params, index=[0]),
                                          sheet_name='network_params_static')
            test_summary_writer.add_sheet(df=pd.DataFrame(self.problem_space_default, index=[0]),
                                          sheet_name='problem_default')

        test_summary_writer.add_sheet(df=pd.DataFrame(all_tests[test_name]['plot_params'], index=[0]),
                                      sheet_name='plot_params')

        timestamp_string = get_timestamp_str()
        print('... Writing test results to sheets: {} and latest'.format(timestamp_string))
        # Overwrite the latest sheet with this data.
        test_summary_writer.add_sheet(test_suite_out, 'latest_data')

        # The output is also saved by timestamp to prevent accidental overwriting in the future
        test_summary_writer.add_rows(test_suite_out, timestamp_string)

def expand_network_params_for_tests(params_for_tests: dict) -> (dict, int):
    if 'cols' not in params_for_tests:
        params_for_tests['cols'] = 1.0
    if 'debug' not in params_for_tests:
        params_for_tests['debug'] = False
    return expand_params_for_tests(params_for_tests=params_for_tests)

def expand_params_for_tests(params_for_tests: dict) -> (dict, int):
    params_for_tests_expanded = params_for_tests

    # Change scalars and arrays to np.ndarray for consistency
    for key, value in params_for_tests.items():
        if isinstance(value, str):
            params_for_tests[key] = np.array([value])
        else:
            if not isinstance(value, np.ndarray):
                if isinstance(value, collections.abc.Sequence):
                    params_for_tests[key] = np.array(value)
                else:
                    params_for_tests[key] = np.array([value])

    num_test_specifications = 1
    for key, value in params_for_tests.items():
        if value.shape[0] > num_test_specifications:
            if num_test_specifications > 1 & (value.shape[0] != num_test_specifications):
                print('Error: The test parameters that vary must all be np arrays of the same length. '
                      'Dict passed was:\n {}'
                      .format(params_for_tests))
                return None, 0
            num_test_specifications = value.shape[0]

    if num_test_specifications != 1:
        for key, value in params_for_tests.items():
            if value.shape[0] != num_test_specifications:
                params_for_tests[key] = np.repeat(value[0], num_test_specifications)

    return params_for_tests_expanded, num_test_specifications


