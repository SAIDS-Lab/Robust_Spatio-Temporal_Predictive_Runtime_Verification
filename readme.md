# Distributionally Robust Predictive Runtime Verification under Spatio-Temporal Specifications

## Introduction
In this project, we demonstrate the Distributionally Robust Runtime Verification Algorithms associated with general CPS under signal-temporal logic (STL) and Multi-agent Systems (MAS) under spatio-temporal reach and escape logic (STREL). The codes contained in this repository are for the paper "Distributionally Robust Runtime Verification Algorithms", which can be found here (to be filled in later once online). Below is the abstract of the paper:

Cyber-physical systems (CPS) designed in simulators, often consisting of multiple interacting agents (e.g. in multi-agent formations), behave differently in the real-world. We would like to verify these systems during runtime when they are deployed. Thus, we propose robust predictive runtime verification (RPRV) algorithms for: (1) general stochastic CPS under signal temporal logic (STL) tasks, and (2) stochastic multi-agent systems (MAS) under spatio-temporal logic
tasks. The RPRV problem presents the following challenges: (1) there may not be sufficient data on the behavior of the deployed CPS, (2) predictive models based on design phase system trajectories may encounter distribution shift during real-world deployment, and (3) the algorithms need to scale to the complexity of MAS and be applicable to spatio-temporal logic tasks. To address these challenges, we assume knowledge of an upper bound on the statistical distance (in terms of an f-divergence) between the trajectory distributions of the system at deployment and design time. We are motivated by our prior work [1, 2] where we proposed an accurate and an interpretable RPRV algorithm for general CPS, which we here extend to the MAS setting and spatio-temporal logic tasks. Specifically, we use a learned predictive model to estimate the system behavior at runtime and robust conformal prediction to obtain probabilistic guarantees by accounting for distribution shifts. Building on [1], we perform robust conformal prediction over the robust semantics of spatio-temporal reach and escape logic (STREL) to obtain centralized RPRV algorithms for MAS. We empirically validate our results in a drone swarm simulator, where we show the scalability of our RPRV algorithms to MAS and analyze the impact of different trajectory predictors on the verification result. To the best of our knowledge, these are the first statistically valid algorithms for MAS under distribution shift.

## Code Structure
We upload the MATLAB setups for both nominal and shifted environment with 5, 7, and 10 agents in [https://drive.google.com/drive/u/0/folders/106AKfbzusk0VrYgOmmjIYY1T1n0sQPgJ](url) that match our experimental setup described in the paper. We collected raw data using these setups. For the raw simulator, one should refer to [https://github.com/lis-epfl/swarmlab](url). For the experiments, we downsampled the trajectories for practical purpopses, and the downsampled trajectories are within each experimental folders. In the auxiliary codes, we contain the training codes for the CNN predictor.

In the lstm folder, we show the codes for the implementation of section 7.1 (Validation of STL RPRV Methods) and section 7.2 (Validation of STREL RPRV Methods): 

0. In `step_0_data_processing.py`, we downsample the raw trajectories and save them in json files (which you do not have to worry about since all the json files are uploaded). 
1. In `step_1_data_analysis.py`, we show the codes on the computation of robust semantics for STREL.
2. In `step_2_predictor_training.py`, we train an LSTM predictor and showcase the prediction examples in nominal and distributionally shifted environments.
3. In `step_3_alpha_calculation.py`, we calculate the normalization constants alphas for the interpretable methods for STREL RPRV in Section 7.2.
4. In `step_4_distribution_shift_computation.py`, we compute the tuning parameter epsilon for Section 7.2.
5. In `step_5_experiments.py`, we conduct the RPRV experiments listed in Section 7.2 on a given number of agents (adjustable in the params file).
6. In `step_6_plot.py`, we make the plots for the paper and analyze the data (including recorded computation times), etc.
7. In `step_7_alpha_calculation_STL.py`, we show the robust semantics calculation of STL and calculate the normalization constants for the interpretable methods for STL RPRV in Section 7.1.
8. In `step_8_distribution_shift_computation_STL.py`, we compute the tuning parameter epsilon for Section 7.1.
9. In `step_9_experiments_STL.py`, we conduct the RPRV experiments listed in Section 7.1 on a single selected agent.
10. In `step_10_plot_STL.py`, we make plots for the paper specifically for Section 7.1.
11. In `step_11_plot_all_agents_togehter.py`, we compare the computation time for different RPRV methods with respect to different number of agents. The plots, however, are not shown in the paper (we instead showed in terms of a table format).
12. In `step_12_compare_predictors.py`, we implement the codes for Section 7.3 where we compare the effect of predictors on the verification results.

Note that apart from the lstm folder, we also have two other folders (aside from auxiliary_codes), cnn and transformer. The purpose of these folders is to contain codes used to compute the alphas and epsilons necessary for the predictor comparison in Section 7.3. Note that the predictor training codes for the transformer are in `transformer/step_2_predictor_training.py`.

Apart from the aforementioned codes, in each folder among lstm, cnn, and transformer, we also contain the file `parameters.py` used to adjust the parameters of running (as we will discuss in the next section) and `data_analysis_test.py`, used to test the computation of STREL robust semantics. 

**Notice that we use direct/indirect methods to refer to accurate/interpretable (Variant I) methods in the codes, and the hybrid method refers to the intepretable (Variant II) method.**

**We also note that in the codes we refer to agent i from the paper as agent i - 1.**

## Replication Procedure

### Section 7.2: 

First to replicate the results in Section 7.2 (with 5 agents), we recommend the following procedure:

1) Set the parameter `num_agents` in `lstm/params.py` to 5.
2) Run `step_0_data_processing.py` up to `step_6_plot.py` in the lstm folder step by step.

The computed alpha values will then be stored in `lstm/alphas/5-agent` along with the computation time. The calculated tuning parameter epsilon will be stored in `lstm/epsilons/5-agent`. The other experiment results are stored in the folder `lstm/experiment_results/5-agent`, which we then use for plotting. The experimental plots are then stored in `lstm/plots/5-agent`.

One can repeat the procedure for L = 7 and L = 10 by adjusting the parameter `num_agents` in `lstm/params.py` and run the aforementioned python files step by step. For plotting the different timing results (which we just showed in table in the paper), please run `step_11_plot_all_agents_together.py` after attaining the results for L = 5, 7, and 10.

### Section 7.1:

To replicate the results in Section 7.1, we recommend the following procedure:

1) Set the parameter `num_agents` in `lstm/params.py` to 5.
2) Run `step_7_alpha_calculation_STL.py` up to `step_10_plot_STL.py` in the lstm folder step by step.

The computed alpha values will then be stored in `lstm/alphas/5-agent/STL`. The calculated tuning parameter epsilon will be stored in `lstm/epsilons/5-agent/STL`. The other experiment results are stored in the folder `lstm/experiment_results/5-agent/STL`, which we then use for plotting. The experimental plots are then stored in `lstm/plots/5-agent/STL`.

### Section 7.3:
To replicate the results in Section 7.3 we reommend the following procedure:

1) Run the aforementioned procedure with 5 agents for section 7.2 with the lstm.
2) Run `step_0_data_processing.py` up to `step_4_distribution_shift_computation.py` in the cnn folder steo by step.
3) Run `step_0_data_processing.py` up to `step_4_distribution_shift_computation.py` in the transformer folder steo by step.
4) Run `step_12_compare_predictors.py` in the lstm folder.

The generated results (in terms of plots and txt files) are then contained in `lstm/comparison_plots`.

### Notes
Note that the load model from the old version of tensorflow was deprecated. Please retrain the predictors to run the experiments (in which case slightly different experimental results can be expected). If you run `step_12_compare_predictors.py` you may also expect different results than those found in the paper since we retrained the LSTM predictor in the updated github repo.

## Contact Information
[Yiqi (Nick) Zhao](https://zhaoy37.github.io/) is a PhD Student for Computer Science at the University of Southern California. For any questions, suggestions, or interests in collaboration, please feel free to contact us at yiqizhao@usc.edu.
