import numpy as np
import pandas as pd

from simulation_scripts.utils.comparisons import compare_binary_signals, compare_tests_to_memories, sum_differences


class ESAMReporter:
    """
    Object to encapsulate the results and information pertaining to a set of recall simulations performed by
    a ESAM network.
    """
    def __init__(self,
                 network_params: dict):
        """
        Initialise the network for a specific simulation implementation based on a dictionary of network parameters and
        simulation parameters.
        :param network_params: a dictionary of parameters that describes the ESAM network.
        """
        # Optional parameters
        if 'f_h_conn_type' not in network_params:
            network_params['f_h_conn_type'] = 'FixedProbability'
        if 'h_f_conn_type' not in network_params:
            network_params['h_f_conn_type'] = 'FixedProbability'
        # Validate that all the network parameters have been provided.
        try:
            self.num_f_neurons = network_params['f']
            self.h_per_memory = network_params['h']
            self.f_h_sparsity = network_params['f_h_sparsity']
            self.h_f_sparsity_e = network_params['h_f_sparsity_e']
            self.h_f_sparsity_i = network_params['h_f_sparsity_i']
            self.epochs_per_recall = network_params['e']
            self.h_threshold = network_params['h_thresh']
            self.f_h_conn_type = network_params['f_h_conn_type']
            self.h_f_conn_type = network_params['h_f_conn_type']
        except KeyError as e:
            print('Error: The following key was missing from the network parameters (net_params) passed to '
                  'ESAMReporter: ', e)
            raise KeyError(e)

        self.initialised = False
        print('\n=========== ESAM Recall Network ============='
              '\n    Num features:                   {}'
              '\n    F to H connection sparsity:     {}'
              '\n      Static F to H conn type?      {}'
              '\n    H to F exitatory sparsity:      {}'
              '\n    H to F inhibitory sparsity:     {}'
              '\n      Static H to F conn type?      {}'
              '\n    Max epochs per recall:          {}'
              '\n    Hidden per memory:              {}'
              '\n    Hidden threshold:               {}'
              '\n'
              .format(self.num_f_neurons,
                      self.f_h_sparsity,
                      self.f_h_conn_type,
                      self.h_f_sparsity_e,
                      self.h_f_sparsity_i,
                      self.h_f_conn_type,
                      self.epochs_per_recall,
                      self.h_per_memory,
                      self.h_threshold))

        # ------------------------------------------------
        # Test definition
        #
        # Create a single row dataframe detailing the network. This will be repeated for each test output.
        dict_as_lists = {'net_param_' + k: [v] for k, v in network_params.items()}
        self.network_df_row = pd.DataFrame.from_dict(dict_as_lists)
        # The ground truth signals corresponding to the labels may be the original signals OR the
        # memory attractor points. The memory attractor points are signals very close to the original memory that
        # the network may resolve to. Therefore, self.ground_truths will always equal either the self.memory_signals
        # or self.memory_attractor_points.
        self.ground_truth_labels = None
        self.ground_truths = None
        self.memory_signals = None
        self.memory_attractor_points = None
        self.use_memory_attractor_points = False

        # ------------------------------------------------
        # Test results
        #
        self.recall_outputs = None
        self.num_reported_results = 0
        # Num_correct is the number of correct results.
        # When divided by the num_reported_results it gives the proportion of correct results.
        self.num_correct = 0

        # ------------------------------------------------
        # Firing activity during the tests
        #
        # Optionally, the reporter can keep track of the proportion of hidden neurons that fired correctly and
        # incorrectly during each epoch. This is useful for debugging and validating thresholds etc.
        # These are np arrays - one proportion value for each test reported.
        # The number of spikes occurring during pattern recognition at each epoch can also be stored.
        # Note that the number of spikes during pattern modulation can be derived from the network config and
        # the numbers of hidden neurons firing. Therefore, this value is not explicitly recorded.
        self.nums_f_h_spikes = None                     # Number of hidden pre-synaptic active connections
        self.nums_correct_h_fired = None                # Number of hidden neurons correctly firing
        self.nums_incorrect_h_fired = None              # Number of hidden neurons incorrectly firing

        # ------------------------------------------------
        # The following dataframes are used to encapsulate all the results.
        self.summary_df = None
        self.summary_detail_df = None

    def add_memory_signals(self, memory_signals: np.ndarray,
                           memory_attractor_points: np.ndarray = None,
                           use_attractors: bool = False):
        """
        Add a list of memory signals to the reporter.
        :param memory_signals:
            The ground truth memory signals known to the network.
        :param memory_attractor_points:
            The memory attractor points (the calculated network's internal representations of the ground truths).
        :param use_attractors:
            A flag indicating whether the network should use the attractor points rather than the memory signals when
            establishing the network's accuracy. By default, the memory signals are used. This flag can be flipped
            later by using the `set_ground_truth_interpretation` method.
        :return:
        """
        self.memory_signals = memory_signals
        self.memory_attractor_points = memory_attractor_points
        self.set_ground_truth_interpretation(use_attractors)

    def set_ground_truth_interpretation(self, use_attractors: bool):
        """
        Define how accuracy will be established by the reporter.
        :param use_attractors:
            Set to True to use the internal attractor representations in accuracy calculations, and False to use the
            ground truth memories.
        :return:
        """
        self.use_memory_attractor_points = use_attractors
        if use_attractors:
            if self.memory_attractor_points is not None:
                self.ground_truths = self.memory_attractor_points
            else:
                print('Memory attractor points not provided by implementation. Using original memories as ground truth')
                self.ground_truths = self.memory_signals
        else:
            self.ground_truths = self.memory_signals
        # To ensure data will be regenerated with the updated interpretation
        self.summary_df = None
        self.summary_detail_df = None

    def add_simulation_result(self, recall_output: np.ndarray, ground_truth_label: int,
                              num_f_h_spikes: np.ndarray, num_correct_h_fired: np.ndarray,
                              num_incorrect_h_fired: np.ndarray) -> bool:
        """
        Record a single simulation result. This comprises the following information:

        * The recall output at each epoch
        * The ground truth for this simulation
        * The number of spikes occurring during each epoch for the pattern recognition step
        * The number of hidden neurons associated with the ground truth that fired for each epoch
        * The number of hidden neurons not associated with the ground truth that fired for each epoch

        Note: the number of spikes occurring during each epoch for the modulation step can be derived
        from the numbers of hidden neurons firing and the network configuration, so is not explicitly
        included in the simulation result.

        :param recall_output:
            The recall signals for each of the epochs during recall.
            A 2-d numpy array containing one recall output (a binary vector of length num_features) for each epoch.
        :param ground_truth_label:
            An integer label indicating the ground truth for this recall.
        :param num_f_h_spikes:
            A numpy array of length num_epochs, dtype integer representing the number of spikes that occurred during the
            pattern recognition step for each epoch.
        :param num_correct_h_fired:
            A numpy array of length num_epochs, dtype integer representing the number of hidden neurons that fired
            correctly on each epoch.
        :param num_incorrect_h_fired:
            A numpy array of length num_epochs, dtype integer representing the number of hidden neurons that fired
            correctly on each epoch.
        :return: True if the test was added successfully, False otherwise.
        """

        if recall_output.shape[0] != self.epochs_per_recall + 1:
            print('Error: ESAMReporter: Number of the signals {} in the recall output did not match the'
                  'expected number {}. This should be the number of cycled per epoch plus one for the original'
                  'signal'.format(recall_output.shape[0], self.epochs_per_recall + 1))
            return False
        if self.ground_truths is None:
            print('Error: ESAMReporter: No memories recorded')
            return False
        if (ground_truth_label > self.ground_truths.shape[0] - 1) or ground_truth_label < 0:
            print('Error: ESAMReporter: Invalid ground truth label passed: {}.'.format(ground_truth_label))
            return False
        if recall_output.shape[1] != self.num_f_neurons:
            print('Error: ESAMReporter: Length of the signals {} in the recall output did not match the'
                  'expected length {}.'.format(recall_output.shape[1], self.num_f_neurons))
            return False

        # Append the result to the existing results
        if not self.initialised:
            # First time around, create a new list
            self.recall_outputs = np.array([recall_output], dtype=int)
            self.ground_truth_labels = np.array([ground_truth_label])

            if num_correct_h_fired is not None:
                self.nums_correct_h_fired = np.array([num_correct_h_fired])
                self.nums_incorrect_h_fired = np.array([num_incorrect_h_fired])
                self.nums_f_h_spikes = np.array([num_f_h_spikes])
            self.initialised = True
        else:
            # Subsequent iterations - append to the existing list
            self.recall_outputs = np.vstack((self.recall_outputs, [recall_output]))
            self.ground_truth_labels = np.append(self.ground_truth_labels, ground_truth_label)
            if num_correct_h_fired is not None:
                self.nums_f_h_spikes = np.vstack((self.nums_f_h_spikes, num_f_h_spikes))
                self.nums_correct_h_fired = np.vstack((self.nums_correct_h_fired, num_correct_h_fired))
                self.nums_incorrect_h_fired = np.vstack((self.nums_incorrect_h_fired, num_incorrect_h_fired))
        self.num_reported_results += 1
        self.__update_recall_stats(recall_output[-1], self.ground_truths[ground_truth_label])
        return True

    def add_simulation_results(self,
                               recall_outputs: np.ndarray,
                               ground_truth_labels: np.ndarray,
                               num_correct_h_fired_all_sims: np.ndarray,
                               num_incorrect_h_fired_all_sims: np.ndarray,
                               num_f_h_spikes_all_sims: np.ndarray) -> bool:
        """
        Add a set of simulation results to the reporter. This method repeatedly calls ``add_simulation_result``.

        :param recall_outputs:
        Numpy array of recall outputs. Length of the primary axis is the number of simulations. Each recall is a
        2-d numpy array containing one recall output (a binary vector of length num_features) for each epoch.
        :param ground_truth_labels:
        Numpy array of ground truth labels. Length of the primary axis is the number of simulations.
        :param num_f_h_spikes_all_sims:
        Numpy array of the number of feature to hidden spikes during the pattern recognition step.
        Length of the primary axis is the number of simulations.
        :param num_correct_h_fired_all_sims:
        Numpy array of proportion of h fired correctly. Length of the primary axis is the number of simulations.
        :param num_incorrect_h_fired_all_sims:
        Numpy array of proportion of h fired incorrectly. Length of the primary axis is the number of simulations.
        :return: True if the test was added successfully, False otherwise.
        """
        if (ground_truth_labels.shape[0] != recall_outputs.shape[0]) \
                or (num_correct_h_fired_all_sims.shape[0] != recall_outputs.shape[0]) \
                or (num_incorrect_h_fired_all_sims.shape[0] != recall_outputs.shape[0]):
            print('Error: ESAMReporter: all the parameters passed to add_test_results should be numpy arrays'
                  'with primary axis of length equal to the number of simulations. Shapes passed were: '
                  '\n       recall_outputs:          {}'
                  '\n       ground_truth_labels:     {}'
                  '\n       props_h_fired_correct:   {}'
                  '\n       props_h_fired_incorrect: {}'
                  .format(recall_outputs.shape, ground_truth_labels.shape,
                          num_correct_h_fired_all_sims.shape, num_incorrect_h_fired_all_sims.shape))
            return False

        for recall_output, ground_truth_label, num_f_h_spikes, num_correct_h_fired, num_incorrect_h_fired in \
                zip(recall_outputs, ground_truth_labels, num_f_h_spikes_all_sims,
                    num_correct_h_fired_all_sims, num_incorrect_h_fired_all_sims):

            if not self.add_simulation_result(recall_output, ground_truth_label, num_f_h_spikes,
                                              num_correct_h_fired, num_incorrect_h_fired):
                return False

        return True

    def get_proportion_correct_recall(self) -> float:
        return round(self.num_correct / self.num_reported_results, 4)

    def get_test_data_averages(self, problem_space, simulator: str = None) -> pd.DataFrame:
        """
        Returns a data frame containing an average of the results for each of the epochs.
        There are (num_epochs+1) rows in the resulting data frame.
        :param problem_space:
        Dictionary containing the data parameters (such as signal sparsity and noise) as these
        are added to the output dataframe.
        :param simulator:
        The name of the simulator (for example 'cpu') is this should be added to the data frame information that is
        returned.
        :return:
        """

        if self.summary_df is not None:
            return self.summary_df

        num_rows = self.epochs_per_recall + 1

        # Epochs: 'epoch'
        epochs = np.full(shape=(num_rows,), fill_value=0, dtype=int)
        # Proportion of hidden neurons that correctly fired, reported for each epoch: 'prop_h_gt'
        av_props_h_gt = np.full(shape=(num_rows,), fill_value=0, dtype=float)
        # Proportion of hidden neurons that incorrectly fired, reported for each epoch: 'prop_h_not_gt'
        av_props_h_not_gt = np.full(shape=(num_rows,), fill_value=0, dtype=float)
        # Length of signal (number of '1's), reported for each epoch: 'signal_length'
        av_signal_lengths = np.full(shape=(num_rows,), fill_value=0, dtype=int)
        # Number of hidden neuron pre-synaptic spikes: 'f_h_spikes'
        av_f_h_spikes = np.full(shape=(num_rows,), fill_value=0, dtype=int)
        # Average number of incorrect and correct results per epoch
        memory_results_correct = np.full(shape=(num_rows,), fill_value=0, dtype=float)
        memory_results_incorrect = np.full(shape=(num_rows,), fill_value=0, dtype=float)

        # We'll use the mean values from the simulations detail
        self.get_test_data_all_simulations()

        for epoch in range(num_rows):
            all_sim_data_for_epoch = self.summary_detail_df.loc[self.summary_detail_df['epoch'] == epoch]
            epochs[epoch] = epoch
            # Get the mean values for each epoch
            memory_results_correct[epoch] = np.sum(all_sim_data_for_epoch['correct'].values) / self.num_reported_results
            memory_results_incorrect[epoch] = \
                np.sum(all_sim_data_for_epoch['incorrect'].values) / self.num_reported_results
            av_props_h_gt[epoch] = np.sum(all_sim_data_for_epoch['prop_h_gt'].values) / self.num_reported_results
            av_props_h_not_gt[epoch] = np.sum(all_sim_data_for_epoch['prop_h_not_gt'].values) / self.num_reported_results
            # The average length of a signal for an epoch during this simulation
            av_signal_lengths[epoch] = \
                int(np.sum(all_sim_data_for_epoch['signal_length'].values) / self.num_reported_results)
            av_f_h_spikes[epoch] = int(np.sum(all_sim_data_for_epoch['f_h_spikes'].values) / self.num_reported_results)

        self.summary_df = pd.DataFrame()
        self.summary_df['epoch'] = epochs
        self.summary_df['prop_correct'] = memory_results_correct
        self.summary_df['prop_h_gt'] = av_props_h_gt
        self.summary_df['prop_h_not_gt'] = av_props_h_not_gt
        self.summary_df['av_signal_length'] = av_signal_lengths
        self.summary_df['f_h_spikes'] = av_f_h_spikes
        # Add the network details (which will be the same for every row) to the dataframe.
        network_df = pd.DataFrame(np.repeat(self.network_df_row.values, repeats=num_rows, axis=0))
        network_df.columns = self.network_df_row.columns

        # Add the simulator if it has been provided
        if simulator is not None:
            network_df['simulator'] = np.repeat(simulator, repeats=num_rows)

        network_df['num_sims'] = np.repeat(self.num_reported_results, repeats=num_rows)

        # Add the data parameters (which will be the same for every row) to the dataframe, if provided.
        if problem_space is not None:
            dict_as_lists = {'data_param_' + k: [v] for k, v in problem_space.items()}
            data_df_row = pd.DataFrame.from_dict(dict_as_lists)
            data_df = pd.DataFrame(np.repeat(data_df_row.values, repeats=num_rows, axis=0))
            data_df.columns = data_df_row.columns
            network_df = pd.concat([network_df, data_df], axis=1)

        self.summary_df = pd.concat([network_df, self.summary_df], axis=1)

        # Calculate the spike data hidden to feature (modulation step)
        # For each row, establish the average number of hidden neurons spiking from the proportions
        self.summary_df['num_spiking_h'] = (self.summary_df['prop_h_gt'] * self.summary_df['net_param_h']) + \
                                           (self.summary_df['prop_h_not_gt'] * (self.summary_df['net_param_h'] *
                                                                                (self.summary_df['data_param_m'] - 1)))
        # Then multiply this with the number of connections per hidden neuron that spikes
        self.summary_df['h_f_spikes'] = self.summary_df['num_spiking_h'] * \
                                        ((self.summary_df['net_param_h_f_sparsity_e'] *
                                          self.summary_df['data_param_s_m'] * self.summary_df['net_param_f']) +
                                         (self.summary_df['net_param_h_f_sparsity_i'] *
                                          (1 - self.summary_df['data_param_s_m']) * self.summary_df['net_param_f']))
        self.summary_df['h_f_spikes'] = self.summary_df['h_f_spikes'].astype(int)

        # Add up the spikes so far and pop in the summary dataframe in the totals columns
        f_h_spikes_so_far = np.full(shape=(num_rows,), fill_value=0)
        h_f_spikes_so_far = np.full(shape=(num_rows,), fill_value=0)
        for i in range(1, num_rows):
            # There is one row per epoch in the summary dataframe.
            pattern_spikes_this_epoch = self.summary_df.loc[[i]].iloc[0]['f_h_spikes']
            f_h_spikes_so_far[i] = f_h_spikes_so_far[i - 1] + pattern_spikes_this_epoch
            mod_spikes_this_epoch = self.summary_df.loc[[i]].iloc[0]['h_f_spikes']
            h_f_spikes_so_far[i] = h_f_spikes_so_far[i - 1] + mod_spikes_this_epoch

        self.summary_df['h_f_spikes_so_far'] = h_f_spikes_so_far
        self.summary_df['f_h_spikes_so_far'] = f_h_spikes_so_far

        return self.summary_df

    def get_test_data_all_simulations(self) -> pd.DataFrame:
        """
        Returns a data frame containing the results for each of the simulations.
        There are (num_epochs+1) * num_simulations rows in the resulting data frame. One row corresponds to one epoch
        of one simulation.
        :return: Pandas dataframe detailing all the results.
        """
        if self.summary_detail_df is not None:
            return self.summary_detail_df

        # There will be one dataframe row generated for each simulation epoch. An additional epoch (epoch 0) refers
        # to the original signal.
        total_rows = self.num_reported_results * (self.epochs_per_recall + 1)

        # The closest memory indices and their similarity scores for every row
        simulations_array = np.full(shape=(total_rows,), fill_value=-1, dtype=int)
        epochs = np.full(shape=(total_rows,), fill_value=-1, dtype=int)

        gt_indexes = np.full(shape=(total_rows,), fill_value=-1, dtype=int)
        gt_diffs = np.zeros(shape=(total_rows,), dtype=int) # np.full(shape=(total_rows,), fill_value=0.0, dtype=float)
        correct = np.zeros(shape=(total_rows,), dtype=int)

        f_h_spikes = np.full(shape=(total_rows,), fill_value=0, dtype=int)
        h_correct_firing = np.full(shape=(total_rows,), fill_value=0, dtype=int)
        h_incorrect_firing = np.full(shape=(total_rows,), fill_value=0, dtype=int)
        num_active_neurons = np.full(shape=(total_rows,), fill_value=-1, dtype=int)

        row = 0

        for simulation, recall_output in zip(range(self.num_reported_results),
                                             self.recall_outputs):

            # Compare all the signal to all the memories across each of the epochs
            epoch_differences_per_mem = compare_tests_to_memories(recall_output,
                                                                  memories=self.ground_truths,
                                                                  comparison_function=sum_differences)
            # Get the ground truth supplied for this simulation
            gt_index = self.ground_truth_labels[simulation]

            # Work out the values for each epoch.
            # - How close is our 'ground truth'?
            # - Was the signal correct for each epoch?
            # - How many hidden neurons fired correctly and incorrectly for each epoch?

            for epoch_num in range(self.epochs_per_recall + 1):
                simulations_array[row] = simulation
                epochs[row] = epoch_num
                gt_indexes[row] = gt_index
                gt_diffs[row] = epoch_differences_per_mem[epoch_num][gt_index]

                if epoch_num != 0:  # No spike data for the initialisation epoch
                    f_h_spikes[row] = self.nums_f_h_spikes[simulation][epoch_num - 1]
                    h_correct_firing[row] = self.nums_correct_h_fired[simulation][epoch_num - 1]
                    h_incorrect_firing[row] = self.nums_incorrect_h_fired[simulation][epoch_num - 1]

                num_active_neurons[row] = np.sum(recall_output[epoch_num])
                row += 1

        correct[np.where(gt_diffs == 0)] = 1
        num_incorrect_h = (self.ground_truths.shape[0] - 1) * self.h_per_memory
        prop_h_correct_firing = h_correct_firing / self.h_per_memory
        prop_h_incorrect_firing = 0
        if num_incorrect_h != 0:  # =0 when there is only one memory so no incorrect hidden
            prop_h_incorrect_firing = h_incorrect_firing / num_incorrect_h

        self.summary_detail_df = pd.DataFrame()

        self.summary_detail_df['sim_num'] = simulations_array
        self.summary_detail_df['epoch'] = epochs
        self.summary_detail_df['correct'] = correct
        self.summary_detail_df['incorrect'] = np.logical_not(correct)

        self.summary_detail_df['signal_length'] = num_active_neurons
        self.summary_detail_df['gt_mem_index'] = gt_indexes
        self.summary_detail_df['gt_mem_diffs'] = gt_diffs
        self.summary_detail_df['f_h_spikes'] = f_h_spikes
        self.summary_detail_df['h_gt'] = h_correct_firing
        self.summary_detail_df['h_not_gt'] = h_incorrect_firing
        self.summary_detail_df['prop_h_gt'] = prop_h_correct_firing
        self.summary_detail_df['prop_h_not_gt'] = prop_h_incorrect_firing
        return self.summary_detail_df

    def __update_recall_stats(self, recall_out: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Update the recall accuracy statistics according to a recall output.

        :param recall_out: Output signal from recall.
        :param ground_truth: Ground truth memory from which the recall signal originated.
        :return: None
        """
        _, num_differences = compare_binary_signals(recall_out, ground_truth)
        if num_differences == 0:
            self.num_correct += 1
