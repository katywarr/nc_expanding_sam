
<p align="center">
  Access this paper: 
  <a href="https://doi.org/10.1162/neco_a_01732">Neural Computation</a>
    |
  <a href="">University of Southampton ePrints (link to be added)</a>
</p>

# Improving Recall in Sparse Associative Memories that use Neurogenesis

Katy Warr ([github](https://github.com/katywarr), [email](mailto:k.s.warr@soton.ac.uk)), 
Jonathon Hare ([github](https://github.com/jonhare), [email](mailto:j.s.hare@soton.ac.uk))
and David Thomas ([email](mailto:d.b.thomas@soton.ac.uk)) 

_Electronics and Computer Science, University of Southampton_


## About

This repository contains the code for our paper 'Improving Recall Accuracy in Sparse Associative Memories that 
use Neurogenesis'. The paper is published in 
<a href="https://doi.org/10.1162/neco_a_01732">Neural Computation (Volume 37, Issue 3)</a> and 
also available through the University of Southampton ePrints page (link to be added in March 2025).

The code demonstrates recall optimisations applied to a subtype of Sparse Associative Memory (SAM) that expands with
memory capacity. This idea is inspired by adult neurogenesis (the creation of new neurons) in the brain; 
a process used to facilitate lifelong learning. 
We therefore refer to this type of network as an *'Expanding Sparse Associative Memory (ESAM)'*.

![Abstract depiction of an ESAM network](ESAM-Abstract-Depiction.png)

Our research explores the characteristics of an ESAM *post learning* that are conducive to optimal recall. Please 
consult the paper for more details.

The code presented is a matrix based implementation written to run efficiently on conventional hardware. Functionally 
equivalent Spiking Neural Network (SNN) implementation could run in a neuromorphic setting.

### Problem Space

A problem space dictionary is passed to the data generation code to describe the nature of the data to be generated.
This dictionary corresponds to **Table 2** in the paper.

* `f`: The length of each signal. This defines the nature of the data that will be automatically 
generated for the test and also the shape of the network.
* `m`: The number of memories. The network will be generated according to these memories.
* `s_m`: Memory sparsity. The sparsity of each memory defined as a proportion of features that are active for the memory.
* `s_n`: The noise sparsity defined as a proportion of features to be flipped (one to zero, or zero to one) 
to convert a memory to a recall signal.

### Network Characterisation

The network description dictionary is used to initialise the binary ESAM. This dictionary describes the characteristics
required of the ESAM network that has undergone learning.

From the network description and the memory signals generated from the problem space, 
a fully trained network will be created. 
This dictionary describes the nature of a learnt ESAM network and corresponds to the parameters 
as described in **Tables 3-7** in the paper. It comprises 
the following keys:

* `f`:  The length of each signal. This defines the nature of the data that will be automatically generated for 
the test and also the shape of the network.
* `h`:  The number of hidden neurons per memory stored. This assumes perfect linear neurogenesis where exactly `h` 
hidden neurons have matured per memory.  
* `f_h_sparsity`: Sparsity of the feature to hidden neuron synaptic connections. See also `f_h_conn_type`.
* `h_f_sparsity_e`: Sparsity of the hidden to feature neuron excitatory synaptic connections. See also `h_f_conn_type`.
* `e`: The number of recall epochs (iterations).
* `h_thresh`: The threshold assigned to every hidden neuron. Each hidden neuron pre-synaptic connection 
carries a weight of `1` and the bias associated with each hidden neuron is `1`. 
During the sub-pattern detection step, a hidden neuron will fire when the number of active pre-synaptic connections 
is equal to or exceeds the threshold.
* `f_h_conn_type`: `FixedNumber` ensures a fixed number of pre-synaptic connections `f * f_h_sparsity * s_m` 
for each hidden neuron. `FixedProbability` means that the probability of a connection from a 
feature neuron to a hidden neuron is `f_h_sparsity * s_m`. 
The latter results in the number of pre-synaptic connections varying according to a 
binomial distribution across the hidden neuron population.

## Getting Started

This code is written with Python3 and runs on CPUs.

### Requirements

The project runs on Python 3.11 and uses the following packages:

* numpy 
* pandas
* scipy
* matplotlib
* seaborn
* openpyxl
* jupyterlab
* tabulate

Package management is accomplished using the [uv](https://docs.astral.sh/uv/) tool which will need to be available on
your local workstation. As explained in [the official uv documentation](https://docs.astral.sh/uv/getting-started/installation/),
a number of straightforward installation options are available (including Homebrew).  
  
After the first clone of this repository run the following command in the root directory. It will automatically create
a new virtual environment in the folder .venv and install all required project dependencies to it.  

```shell
uv sync
```

### Running Jupyter Lab


The virtual environment will include a Python 3 runtime at the version specified in the `.python-version` file in this
directory. Activate the virtual environment in your current terminal as follows:

```shell
source .venv/bin/activate
```

With the virtual environment active, Jupyter can be started by switching to the `examples` folder and running the 
command `jupyter lab`:

```shell
cd examples
jupyter lab
```

It is also possible to skip the activation of the virtual environment and just use the command 
`uv run jupyter lab` from the `examples` folder.

### Running the Network 

The folder `examples` contains Jupyter notebooks to experiment with the network and to recreate the specific 
experiments in the paper.

To experiment with different ESAM problem spaces and network definitions, run the Jupyter notebook 
`1 - ESAM_tutorial.ipynb`. This notebook
provides example code to generate data, create an ESAM network, and perform associative 
memory recall. This is a good place to get started. It also provides test diagnostics over multiple simulations.

## Reproducing the Experiments in the Paper

Each of the figures generated in the paper is associated with two Jupyter notebooks. 

* Run: The data generation notebook recreates the 
data and stores the results in Excel spreadsheets under the `examples\experiment_results` folder. 
* Plot: The data plotting 
notebook reads from this data and plots the relevant figure(s). By default, this notebook will 
re-plot the pre-generated paper data, so there is no need to re-run the experiment if you would just like to 
generate a plot. The generated figures from the paper are also in the `figures` folder.

Each Excel file contains several sheets:

* `paper_data`: containing the pre-generated data used for the figures in the paper. By default, the plot notebooks use 
this data.
* `network_params_static` and `problem_dafault`: summarises the network parameters and base problem definition for 
experiments where the network remains static and the problem definition changes on the x-axis. 
For example, this could be testing a static network definition with varying numbers of memories (`m`).
* `problem_static` and `network_params_default`:  summarises the problem definition and base network for experiments 
where the problem remains static and the network changes on the x-axis. For example, this could be testing a 
static problem definition with varying connection sparsity.
* `plot_params`: a few parameters are stored to help in plotting the test.

When the tests are re-run using the data generation notebook, the new results are stored in a timestamped sheet
(to prevent overwriting) and also in the `latest_data` sheet (which is overwritten). Therefore, 
the latest data is always saved to the sheet 
named ``latest`` and the ``paper_data`` sheet remains untouched. This makes it easy to plot the 
latest data results; just change `paper_data` to `latest_data` in the relevant Jupyter notebook.

### Figures 9 and 10

* **Run:** `2 - Experiment_A_Sub_pattern - Run.ipynb`
* **Plot:** `3 - Experiment_A_Sub_pattern - Plot Figs 9 and 10.ipynb`

### Figures 11, 12, and 13

* **Run:** `4 - Experiment_B_sub_pattern - Run.ipynb`
* **Plot:** `5 - Experiment_B_sub_pattern - Plot Figs 11 13 and 13.ipynb`

### Figure 14

* **Run:** `6 - Experiment_B_modulation - Run.ipynb`
* **Plot:** `7 - Experiment_B_modulation - Plot Fig 14.ipynb`

### Figure 15

* **Run:** `8 - Experiment_B_all_opts - Run.ipynb`
* **Plot:** `9 - Experiment_B_all_opts - Plot Fig 15.ipynb`

##  Code Structure

The sources root directory is `src`:
* ``src`` contains the core ESAM implementation. Specifically: 
  * `binary_esam.py`: the ESAM network class.
  * `esam_reporter.py`: a separate class to encapsulate results. There is one `esam_reporter` per network.
* ``src\impl`` contains the ESAM implementation classes.
* ``src\data_generation`` contains code to generate memory signals and noisy variants.
* ``src\simulation_scripts`` helper scripts to run and plot the results of tests. This makes it easier to run 
tests over a number of simulations, save the results, and produce plots.
* ``src\simulation_scripts\utils`` contains utility functions for the simulations including code to generate 
the optimum threshold $\theta$ for a test (as discussed in **section 6.2** of the paper)
and code to calculate the theoretical recall probability (as discussed in **section 8** of the paper).
Because these calculations are cpu intensive, the probabilities in `hyperparameter_data\hidden_neuron_activations.xlsx`
and only re-generated as and when they are required.

## Acknowledgements

This work was supported by the UK Research and Innovation (UKRI) Engineering and
Physical Sciences Research Council (EPSRC).

The authors acknowledge the use of the IRIDIS High Performance Computing Facility, 
and associated support services at the University of Southampton, in the 
completion of this work.



