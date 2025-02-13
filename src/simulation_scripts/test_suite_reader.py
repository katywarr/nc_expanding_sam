import numpy as np
import pandas as pd
from simulation_scripts.utils.excel_handler import ExcelHandler
import os
from simulation_scripts.utils.probabilities import prob_f_h_carries_signal_fn, prob_conns_fn, prob_activation_fn, prob_firing_fn, \
    prob_correct_pattern_retrieval

def esam_topology(f, m, s_m, f_h_sparsity, h, h_f_sparsity_e, h_f_sparsity_i):
    max_conns = f * h * m
    conns_f_h = max_conns * s_m * f_h_sparsity
    conns_h_f_e = max_conns * s_m * h_f_sparsity_e
    conns_h_f_i = max_conns * (1 - s_m) * h_f_sparsity_i
    topology = {
        'conns_f_h': conns_f_h,
        'conns_h_f_e': conns_h_f_e,
        'conns_h_f_i': conns_h_f_i,
        'conns_h_f': conns_h_f_e + conns_h_f_i,
        'conns_total': conns_h_f_e + conns_h_f_i + conns_f_h,
    }
    return topology


def get_topology_data_for_test(test_dir: str, test_file: str, data_sheet: str, variable_column: str):
    """
    Creates topology data for the associated data in the test file specified.
    Topology data is only created for each row that has a unique value in the test data column specified by
    `variable_column`. `variable_column` is essentially the x-axis if the data were to be plotted.
    :param test_dir:
        Test directory containing the data file.
    :param test_file:
        Test file within the directory (minus '.xlsx' extension).
    :param data_sheet:
        Excel sheet name containing the data.
    :param variable_column:
        Variable column. Rows will be selected based on unique values in this column.
    :return:
        Pandas data set containing topology data and associated sample test data.
    """
    print('Generating topology data for test: {} and datasheet: {}'
          .format(test_dir + os.sep + test_file, data_sheet))

    reader = ExcelHandler(test_dir, test_file)

    original_test_data_df = reader.read_sheet(data_sheet)

    test_name_np = np.repeat([test_file], repeats=original_test_data_df.shape[0], axis=0)
    original_test_data_df['test'] = test_name_np

    # Select each of the unique values along the x-axis
    x_column_variation = np.unique(original_test_data_df[variable_column])

    # And filter the columns that we want
    reduced_test_data_df = pd.DataFrame()
    for row_num, variable in zip(range(len(x_column_variation)), x_column_variation):
        df_row = pd.DataFrame([original_test_data_df.loc[original_test_data_df[variable_column] == variable].iloc[0]])
        df_row_filtered = df_row[['net_param_f', 'net_param_h', 'data_param_m', 'net_param_f_h_sparsity',
                                  'net_param_h_f_sparsity_e', 'net_param_h_f_sparsity_i', 'data_param_s_m']].copy()
        reduced_test_data_df = pd.concat([reduced_test_data_df, df_row_filtered])

    topology = esam_topology(f=reduced_test_data_df['net_param_f'],
                             m=reduced_test_data_df['data_param_m'],
                             s_m=reduced_test_data_df['data_param_s_m'],
                             f_h_sparsity=reduced_test_data_df['net_param_f_h_sparsity'],
                             h=reduced_test_data_df['net_param_h'],
                             h_f_sparsity_e=reduced_test_data_df['net_param_h_f_sparsity_e'],
                             h_f_sparsity_i=reduced_test_data_df['net_param_h_f_sparsity_i'])

    topology_df = pd.DataFrame.from_dict(topology)

    return pd.concat([reduced_test_data_df, topology_df], axis=1)

class TestSuiteReader:
    """
    Convenience class to read the data pertaining to previously run tests.
    This class also accesses theoretical results for comparison with the empirical data.
    """

    def __init__(self,
                 test_dir: str,
                 test_suite_type: str,
                 tests: dict,
                 test_data_sheet: str = 'latest_data',
                 plot_params_sheet: str = 'plot_params'
                 ):
        """
        Initialise the TestSuiteReader based on the data within an Excel file. This includes details of the tests
        run (the problem space and network description) and the test results.

        :param test_dir:
            A string describing the directory containing the Excel files that contain the data.
        :param test_suite_type:
            The test suite is either of type 'vary_network', in which case the tests have stable problem places and
            one or more network attributes is varied, or 'vary_problem', in which case the tests have a stable
            networks and one or more problem space attributes is varied.
        :param tests:
            A dictionary of the tests.
            The keys are strings indicating the file name in the test suite without the '.xlsx' extension. The values
            are the readable names of the tests (useful for plotting).
        :param test_data_sheet:
            The name of the sheet in each file where the tests are located. This defaults to 'latest_data'. The name of
            the sheet must be the same across each of the files.
        :param plot_params_sheet:
            The name of the sheet in each file where the plot_params are located. This defaults to 'plot_params'.
            This sheet provides hints regarding how the data should be plotted.
        """
        # Check that the base folder exists
        if not os.path.exists(test_dir):
            print('Error: TestSuitePlotter - Directory containing the test data does not exist: {} '
                  '\nCurrent working directory is: {}'
                  .format(test_dir, os.getcwd()))
            return

        self.test_dir = test_dir        # Needed later for lazy getting of theoretical probabilities
        self.test_suite_type = test_suite_type
        if test_suite_type == 'vary_network':
            self.vary_col_prefix = 'net_param_'
            default_sheet_name = 'network_default'
        else:
            if test_suite_type == 'vary_problem':
                self.vary_col_prefix = 'data_param_'
                default_sheet_name = 'problem_default'
            else:
                raise ValueError('Error: test suite type of \'vary_network\' or \'vary_problem\' must be passed to the'
                                 ' test suite reader.')
        self.test_suite_type = test_suite_type
        self.test_files = list(tests.keys())
        self.tests = tests
        # The following lists will store one Pandas dataframe for each test file.
        self.df_per_test_plot_params = []
        self.df_per_test_default_params = []
        # The following dataframe stores all the data, with an additional 'test' column to indicate the test.
        # This is a useful format for plotting.
        self.df_all_tests_data = pd.DataFrame()

        # Check each of the test files contains data and populate lists with the required data.
        print('tests are : ', self.test_files)
        print('test dir: ', test_dir)
        print('Using the following pre-generated data:\n')
        print('  File                                                            Data Sheet')
        print('  ---------------------------------------------------------------------------')

        for test_file, test_name in tests.items():
            # Read the data for this test from the Excel file.
            # The test_excel_data will handle any errors and throw exceptions. Let these trickle back to the caller.
            test_excel_data = ExcelHandler(file_dir=test_dir, file_name=test_file)
            excel_file = test_dir + os.sep + test_file + '.xlsx'
            print('  {:60s}    {:10s}'.format(excel_file, test_data_sheet))

            # Test plot parameters
            self.df_per_test_plot_params.append(test_excel_data.read_sheet(sheet_name=plot_params_sheet).T.to_dict()[0])

            # Test Defaults
            test_defaults = test_excel_data.read_sheet(sheet_name=default_sheet_name).T.to_dict()[0]
            if self.test_suite_type == 'vary_problem':
                # The number of epochs is not in the problem space defaults. Add this manually from the static network
                # data. Not all that elegant, but it works later.
                static_network = test_excel_data.read_sheet(sheet_name='network_params_static').T.to_dict()[0]
                test_defaults['e'] = static_network['e']
            self.df_per_test_default_params.append(test_defaults)

            # Test Data
            test_results = test_excel_data.read_sheet(sheet_name=test_data_sheet)
            # For the single data frame, add the name of the test to the row and concatenate
            test_results['test'] = test_name
            # Remove epoch 0 - this is the test definition epoch which is not required
            self.df_all_tests_data = pd.concat([self.df_all_tests_data,
                                                test_results.copy().loc[test_results['epoch'] != 0]])

        print('\nTestSuiteResults successfully initialised for tests {}'.format(self.test_files))

    def get_tests_data(self) -> pd.DataFrame:
        """
        Return the tests data for the tests associated with this object
        :return:
        """
        return self.df_all_tests_data

    def get_theory_empirical_probs(self) -> list:
        """
        Return the empirical results along with the theoretical data for examination/plotting.
        The empirical data includes the proportion of hidden neurons firing correctly/incorrectly and the proportion of
        correct results. The theoretical data comprises the expected probabilities (proportions) based on the
        equations presented in the paper.
        :return:
        """

        theory_data = self.__get_theoretical_probabilities()

        tests_filtered = []
        for (_, test_name), theory_df in zip(self.tests.items(), theory_data):
            # Empirical data
            # Filter on the test results for this test.
            test_df = self.df_all_tests_data.copy().loc[self.df_all_tests_data['test'] == test_name]
            # It's just the first epoch we are interested in, filter on this one
            test_df_first_epoch = test_df.copy().loc[test_df['epoch'] == 1]
            test_df_first_epoch['Theory or empirical'] = np.repeat('empirical', test_df_first_epoch.shape[0])
            # Add new probability columns so that they all match perfectly (required for concat and plotting below)
            test_df_first_epoch['h_correct_prob'] = test_df_first_epoch['prop_h_gt']
            test_df_first_epoch['h_incorrect_prob'] = test_df_first_epoch['prop_h_not_gt']
            test_df_first_epoch['prob_correct'] = test_df_first_epoch['prop_correct']
            test_df_first_epoch['test_name'] = test_name

            # Theoretical data
            test_df_theory_new = theory_df.copy()
            test_df_theory_new['Theory or empirical'] = np.repeat('theory', test_df_theory_new.shape[0])

            # Combine and add to the list for plotting
            test_all = pd.concat([test_df_first_epoch, test_df_theory_new])
            tests_filtered.append(test_all)

        return tests_filtered

    def __get_theoretical_probabilities(self):

        df_theory_list = []     # A list of data frames, one per test, containing theoretical calculations

        # Pull out the relevant test description variations
        for plot_params, network_params, (_, test_name), test_file_name in zip(self.df_per_test_plot_params,
                                                                   self.df_per_test_default_params,
                                                                   self.tests.items(),
                                                                   self.test_files):
            # The test_excel_data will handle any errors and throw exceptions. Let these trickle back to the caller.
            test_excel_data = ExcelHandler(file_dir=self.test_dir, file_name=test_file_name)
            if test_excel_data.check_sheet_name('theory'):
                print('Reading theoretical probabilities for test {} from file.'.format(test_file_name))
                df_theory = test_excel_data.read_sheet(sheet_name='theory')
            else:
                print('Calculating theoretical probabilities for test {}.'.format(test_file_name))
                # Extract the data for this test
                test_df = self.df_all_tests_data.copy().loc[self.df_all_tests_data['test'] == test_name]
                # For this test, get the unique values from our variable column and put in a numpy array
                x_col = self.vary_col_prefix + plot_params['variable_column']
                variable_data = np.unique(test_df[x_col])
                # Create a data frame to represent each of the variations for this test. One row per variation.
                num_rows = len(variable_data)
                df_theory = pd.DataFrame()
                # Copy the data required for plotting/future calculations
                for row_num, variable in zip(range(num_rows), variable_data):
                    df_row = pd.DataFrame([test_df.loc[test_df[x_col] == variable].iloc[0]])
                    df_row_filtered = df_row.copy()
                    signal_fn = prob_f_h_carries_signal_fn(s_m=df_row_filtered['data_param_s_m'].iloc[0],
                                                           s_n=df_row_filtered['data_param_s_n'].iloc[0])
                    l_m = round(df_row_filtered['data_param_s_m'].iloc[0] * df_row_filtered['net_param_f'].iloc[0])
                    conn_fn = prob_conns_fn(length=l_m,
                                            sparsity=df_row_filtered['net_param_f_h_sparsity'].iloc[0],
                                            conn_type=df_row_filtered['net_param_f_h_conn_type'].iloc[0])
                    act_fn = prob_activation_fn(max_connections=l_m, signal_fn=signal_fn, connection_fn=conn_fn)
                    firing_fn = prob_firing_fn(max_conns=l_m, activation_fn=act_fn)
                    print('   Calculating theoretical probabilities for {} = {} of {}'
                          .format(x_col, variable, variable_data))

                    h_correct_prob = firing_fn(recall_type='correct',
                                               theta=df_row_filtered['net_param_h_thresh'].iloc[0])
                    h_incorrect_prob = firing_fn(recall_type='incorrect',
                                                 theta=df_row_filtered['net_param_h_thresh'].iloc[0])
                    df_row_filtered['h_correct_prob'] = h_correct_prob
                    df_row_filtered['h_incorrect_prob'] = h_incorrect_prob
                    inhib_value = df_row_filtered['net_param_h_f_sparsity_i'].iloc[0]
                    if inhib_value != 0:
                        inhib = True
                        if inhib_value != 1:
                            print('Warning: Setting inhibition {} to 1 for theoretical calculations.'
                                  .format(inhib_value))
                            df_row_filtered.loc[0, 'net_param_h_f_sparsity_i'] = 1
                            # df_row_filtered['net_param_h_f_sparsity_i'].iloc[0] = 1
                    else:
                        inhib = False
                    df_row_filtered['prob_correct'] = (
                        prob_correct_pattern_retrieval(m=df_row_filtered['data_param_m'].iloc[0],
                                                       h=df_row_filtered['net_param_h'].iloc[0],
                                                       prob_h_firing_correct=h_correct_prob,
                                                       prob_h_firing_incorrect=h_incorrect_prob,
                                                       inhibition=inhib))
                    df_theory = pd.concat([df_theory, df_row_filtered])

                # Save the data for next time
                test_excel_data.add_sheet(df=df_theory, sheet_name='theory')

            # Add the data to the list of theory dataframes
            df_theory_list.append(df_theory)

        return df_theory_list
