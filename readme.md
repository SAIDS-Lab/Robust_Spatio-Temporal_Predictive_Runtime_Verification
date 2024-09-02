# Distributionally Robust Predictive Runtime Verification under Spatio-Temporal Specifications

## Introduction
In this project, we demonstrate the Distributionally Robust Runtime Verification Algorithms associated with general CPS under signal-temporal logic (STL) and Multi-agent Systems (MAS) under spatio-temporal reach and escape logic (STREL). The codes contained in this repository are for the paper "Distributionally Robust Runtime Verification Algorithms", which can be found here (to be filled in later once online). Below is the abstract of the paper:

To be filled in once online.

Image to be filled in once online.

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
12. In `step_12_compare_predictors.py`, we implement the codes for section 7.3 where we compare the effect of predictors on the verification results.

Note that apart from the lstm folder, we also have two other folders (aside from auxiliary_codes), cnn and transformer. The purpose of these folders is to contain codes used to compute the alphas and epsilons necessary for the predictor comparison in section 7.3. Note that the predictor training codes for the transformer are in `transformer/step_2_predictor_training.py`.

Apart from the aforementioned codes, in each folder among lstm, cnn, and transformer, we also contain the file `parameters.py` used to adjust the parameters of running (as we will discuss in the next section) and `data_analysis_test.py`, used to test the computation of STREL robust semantics. 

Notice that we use direct/indirect methods to refer to accurate/interpretable methods in the codes.

## Replication Procedure

Todo: where to find results.