import numpy as np
import params
import keras
import time
import json
import step_0_data_processing
import step_2_predictor_training
import step_5_experiments
import step_7_alpha_calculation_STL
import step_8_distribution_shift_computation_STL


def compute_worst_case_robustness_indirect(history, prediction_region):
    worst_case_predicate_ground_collision = dict()
    worst_case_predicate_obstacle_avoidance = dict()
    worst_case_predicate_goal_reaching = dict()
    for tau in range(0, params.T + 1):
        if tau <= params.current_time:
            worst_case_predicate_ground_collision[tau] = step_7_alpha_calculation_STL.evaluate_ground_collision_avoidance(history, tau)
            worst_case_predicate_obstacle_avoidance[tau] = step_7_alpha_calculation_STL.evaluate_obstacle_avoidance(history, tau)
            worst_case_predicate_goal_reaching[tau] = step_7_alpha_calculation_STL.evaluate_goal_reaching(history, tau)
        else:
            x_hat = history[tau]
            worst_case_predicate_ground_collision[tau] = step_5_experiments.find_optimal_ground(x_hat, prediction_region[tau])
            worst_case_predicate_obstacle_avoidance[tau] = step_5_experiments.find_optimal_obstacles(x_hat, prediction_region[tau])
            worst_case_predicate_goal_reaching[tau] = step_5_experiments.find_optimal_goal(x_hat, prediction_region[tau])
    # Find the final worst case robustness.
    robustness_1 = float("inf")
    for t in range(0, params.T + 1):
        robustness_1 = min(robustness_1, min(worst_case_predicate_ground_collision[t], worst_case_predicate_obstacle_avoidance[t]))
    robustness_2 = -float("inf")
    for t in range(0, params.T + 1):
        robustness_2 = max(robustness_2, worst_case_predicate_goal_reaching[t])
    return min(robustness_1, robustness_2)


def compute_worst_case_robustness_hybrid(history, alphas_hybrid, c_value):
    worst_case_predicate_ground_collision = dict()
    worst_case_predicate_obstacle_avoidance = dict()
    worst_case_predicate_goal_reaching = dict()
    for tau in range(0, params.T + 1):
        if tau <= params.current_time:
            worst_case_predicate_ground_collision[tau] = step_7_alpha_calculation_STL.evaluate_ground_collision_avoidance(history, tau)
            worst_case_predicate_obstacle_avoidance[tau] = step_7_alpha_calculation_STL.evaluate_obstacle_avoidance(history, tau)
            worst_case_predicate_goal_reaching[tau] = step_7_alpha_calculation_STL.evaluate_goal_reaching(history, tau)
        else:
            x_hat = history[tau]
            worst_case_predicate_ground_collision[tau] = x_hat[2] - params.ground_height - c_value * alphas_hybrid["ground_collision"][tau]
            worst_case_predicate_obstacle_avoidance[tau] = step_5_experiments.closest_euclidean_distance_to_obstacles(x_hat) - c_value * alphas_hybrid["obstacle_avoidance"][tau]
            worst_case_predicate_goal_reaching[tau] = x_hat[0] - params.goal_location - c_value * alphas_hybrid["goal_reaching"][tau]
    # Find worst case robustness for final specification.
    robustness_1 = float("inf")
    for t in range(0, params.T + 1):
        robustness_1 = min(robustness_1, min(worst_case_predicate_ground_collision[t], worst_case_predicate_obstacle_avoidance[t]))
    robustness_2 = -float("inf")
    for t in range(0, params.T + 1):
        robustness_2 = max(robustness_2, worst_case_predicate_goal_reaching[t])
    return min(robustness_1, robustness_2)


def calculate_indirect_nonconformity_2023(trajectories, predicted_trajectories, tau):
    nonconformity_scores = []
    indices = list(sorted(trajectories.keys()))
    for index in indices:
        x = trajectories[index][tau]
        x_hat = predicted_trajectories[index][tau]
        distance = ((x[0] - x_hat[0]) ** 2 + (x[1] - x_hat[1]) ** 2 + (x[2] - x_hat[2]) ** 2) ** (1 / 2)
        nonconformity_scores.append(distance)
    return nonconformity_scores


def main():
    print("=== Experiment Starts ===")
    np.random.seed(params.my_seed)

    print("=== Load Ground Trajectories ===")
    # First, load the calibration trajectories.
    trajectories = step_0_data_processing.load_trajectories()
    # Load only the calibration trajectories.
    calib_trajectories_all = dict()
    for i in range(params.num_train + 1, params.num_histories + 1):
        calib_trajectories_all[i] = trajectories[i]
    # Then, load the test trajectories.
    test_trajectories_all = step_0_data_processing.load_trajectories(shifted=True)
    print("=== Finished Loading Ground Trajectories ===")
    print()

    print("=== Loading the LSTM Model ===")
    trained_lstm_models = dict()
    for a in range(params.num_agents):
        trained_lstm_models[a] = dict()
        for s in range(3):
            trained_lstm_models[a][s] = keras.models.load_model(
                f"predictors/{params.num_agents}-agent/lstm/lstm_model_agent_{a}_state_{s}.keras")
    print("=== Finished Loading the LSTM Model ===")
    print()

    print("=== Load the Norm for the LSTM Model ===")
    with open(f'predictors/{params.num_agents}-agent/lstm/norm.txt', 'r') as f:
        lstm_norm = float(f.read())
    print("=== Finished Loading the Norm for the LSTM Model ===")
    print()

    print("== Produce Predicted Trajectories for the calibration data. ===")
    predicted_calib_trajectories_all = step_2_predictor_training.generate_pred_trajectories(trained_lstm_models,
                                                                                            calib_trajectories_all,
                                                                                            lstm_norm)
    print("=== Finished Producing Predicted Trajectories for the calibration data. ===")
    print()
    print("=== Produce Predicted Trajectories for the test data. ===")
    predicted_test_trajectories_all = step_2_predictor_training.generate_pred_trajectories(trained_lstm_models,
                                                                                           test_trajectories_all,
                                                                                           lstm_norm)
    print("=== Finished Producing Predicted Trajectories for the test data. ===")
    print()

    print("=== Extract trajectories for the ego agent ===")
    for key in calib_trajectories_all.keys():
        calib_trajectories_all[key] = calib_trajectories_all[key][params.ego_agent]
    for key in test_trajectories_all.keys():
        test_trajectories_all[key] = test_trajectories_all[key][params.ego_agent]
    for key in predicted_calib_trajectories_all.keys():
        predicted_calib_trajectories_all[key] = predicted_calib_trajectories_all[key][params.ego_agent]
    for key in predicted_test_trajectories_all.keys():
        predicted_test_trajectories_all[key] = predicted_test_trajectories_all[key][params.ego_agent]
    print("=== Finished Extracting trajectories for the ego agent ===")
    print()

    print("=== Loading Epsilons ===")
    with open(f"epsilons/{params.num_agents}-agent/STL/final_epsilon.txt", "r") as f:
        epsilon = float(f.read())
    print("=== Finished Loading Epsilons ===")
    print()

    print("=== Loading Alphas ===")
    with open(f'alphas/{params.num_agents}-agent/STL/alphas_indirect.json', 'r') as f:
        alphas_indirect = json.load(f)
        new_alphas_indirect = dict()
        for tau in alphas_indirect:
            new_alphas_indirect[int(tau)] = alphas_indirect[tau]
        alphas_indirect = new_alphas_indirect
    with open(f'alphas/{params.num_agents}-agent/STL/alphas_hybrid.json', 'r') as f:
        alphas_hybrid = json.load(f)
        new_alphas_hybrid = dict()
        for predicate in alphas_hybrid:
            new_alphas_hybrid[predicate] = dict()
            for tau in alphas_hybrid[predicate]:
                new_alphas_hybrid[predicate][int(tau)] = alphas_hybrid[predicate][tau]
        alphas_hybrid = new_alphas_hybrid
    print("=== Finished Loading Alphas ===")
    print()

    print("=== Start the main experiment procedure ===")
    # Calculate delta_tilde.
    delta_n = step_5_experiments.calculate_delta_n(params.delta, params.num_calibration_each_trial, step_5_experiments.f_divergence, epsilon)
    delta_tilde = step_5_experiments.calculate_delta_tilde(delta_n, step_5_experiments.f_divergence, epsilon)
    print("Delta tilde is:", delta_tilde)
    direct_vanilla_coverages = []
    direct_robust_coverages = []
    indirect_vanilla_coverages = []
    indirect_robust_coverages = []
    hybrid_vanilla_coverages = []
    hybrid_robust_coverages = []
    direct_times = []
    direct_test_times = []
    indirect_times = []
    indirect_test_times = []
    hybrid_times = []
    hybrid_test_times = []
    for experiment_num in range(params.num_experiments):
        print(">>>>> Conducting Experiment:", experiment_num + 1)
        # First, sample trajectories from calibration trajectories all.
        random_indices_calib_trajectories = np.random.choice(list(calib_trajectories_all.keys()),
                                                             params.num_calibration_each_trial, replace=False)
        calib_trajectories = dict()
        for index in random_indices_calib_trajectories:
            calib_trajectories[index] = calib_trajectories_all[index]
        calib_predicted_trajectories = dict()
        for index in random_indices_calib_trajectories:
            calib_predicted_trajectories[index] = predicted_calib_trajectories_all[index]
        # Sample trajectories from the test trajectories.
        random_indices_test_trajectories = np.random.choice(list(test_trajectories_all.keys()),
                                                            params.num_test_each_trial, replace=False)
        test_trajectories = dict()
        for index in random_indices_test_trajectories:
            test_trajectories[index] = test_trajectories_all[index]
        test_predicted_trajectories = dict()
        for index in random_indices_test_trajectories:
            test_predicted_trajectories[index] = predicted_test_trajectories_all[index]

        # Perform the direct method.
        print("=== Conducting the direct method. ===")
        # Timing for the direct method.
        direct_start_time = time.time()
        direct_nonconformity_list = step_8_distribution_shift_computation_STL.calculate_direct_nonconformity(calib_trajectories, calib_predicted_trajectories)
        direct_nonconformity_list.sort()
        # Perform vanilla conformal prediction.
        direct_nonconformity_list.append(float("inf"))
        # Find c.
        p = int(np.ceil((len(direct_nonconformity_list)) * (1 - params.delta)))
        c = direct_nonconformity_list[p - 1]
        # Perform robust conformal prediction.
        direct_nonconformity_list = direct_nonconformity_list[:-1]
        p_tilde = int(np.ceil(len(direct_nonconformity_list) * (1 - delta_tilde)))
        c_tilde = direct_nonconformity_list[p_tilde - 1]
        direct_time = time.time() - direct_start_time
        direct_times.append(direct_time)
        print("Calibration Time for the Direct Method:", direct_time, "seconds.")
        # Save the data for plotting.
        if experiment_num == 0:
            with open(f"experiment_results/{params.num_agents}-agent/STL/direct_nonconformity_list.json", "w") as file:
                json.dump(direct_nonconformity_list, file)
            with open(f"experiment_results/{params.num_agents}-agent/STL/c_direct.txt", "w") as file:
                file.write(str(c))
            with open(f"experiment_results/{params.num_agents}-agent/STL/c_tilde_direct.txt", "w") as file:
                file.write(str(c_tilde))
        # Calculate the coverage for the direct methods.
        direct_ground_robustnesses = []
        direct_worst_robustnesses_vanilla = []
        direct_worst_robustnesses_robust = []
        direct_coverage_count_vanilla = 0
        direct_coverage_count_robust = 0
        direct_test_start_time = time.time()
        for index in random_indices_test_trajectories:
            history = test_trajectories[index]
            direct_ground_robustness = step_7_alpha_calculation_STL.evaluate_final_robustness(history)
            pred_history = test_predicted_trajectories[index]
            pred_robustness = step_7_alpha_calculation_STL.evaluate_final_robustness(pred_history)
            direct_worst_robustness_vanilla = pred_robustness - c
            direct_worst_robustness_robust = pred_robustness - c_tilde
            if direct_ground_robustness >= direct_worst_robustness_vanilla:
                direct_coverage_count_vanilla += 1
            if direct_ground_robustness >= direct_worst_robustness_robust:
                direct_coverage_count_robust += 1
            direct_ground_robustnesses.append(direct_ground_robustness)
            direct_worst_robustnesses_vanilla.append(direct_worst_robustness_vanilla)
            direct_worst_robustnesses_robust.append(direct_worst_robustness_robust)
        direct_test_time = time.time() - direct_test_start_time
        direct_test_times.append(direct_test_time)
        print("Test time for the direct method", direct_test_time, "seconds.")
        # Save the data.
        if experiment_num == 0:
            with open(f"experiment_results/{params.num_agents}-agent/STL/direct_ground_robustnesses.json", "w") as file:
                json.dump(direct_ground_robustnesses, file)
            with open(f"experiment_results/{params.num_agents}-agent/STL/direct_worst_robustnesses_vanilla.json", "w") as file:
                json.dump(direct_worst_robustnesses_vanilla, file)
            with open(f"experiment_results/{params.num_agents}-agent/STL/direct_worst_robustnesses_robust.json", "w") as file:
                json.dump(direct_worst_robustnesses_robust, file)
        direct_vanilla_coverage = direct_coverage_count_vanilla / params.num_test_each_trial
        direct_robust_coverage = direct_coverage_count_robust / params.num_test_each_trial
        direct_vanilla_coverages.append(direct_vanilla_coverage)
        direct_robust_coverages.append(direct_robust_coverage)
        print("Direct Coverage Vanilla CP:", direct_vanilla_coverage)
        print("Direct Coverage Robust CP:", direct_robust_coverage)
        print("=== Finished conducting the direct method. ===")

        # Perform the indirect method.
        print("===Conducting the indirect method.===")
        # Time for the indirect method.
        indirect_start_time = time.time()
        indirect_nonconformity_list = step_8_distribution_shift_computation_STL.calculate_indirect_nonconformity(calib_trajectories, calib_predicted_trajectories, alphas_indirect)
        indirect_nonconformity_list_2023 = calculate_indirect_nonconformity_2023(calib_trajectories, calib_predicted_trajectories, params.indirect_illustration_variant_1_tau)
        indirect_nonconformity_list.sort()
        indirect_nonconformity_list_2023.sort()
        # Perform vanilla conformal prediction.
        indirect_nonconformity_list.append(float("inf"))
        # Find c.
        p = int(np.ceil((len(indirect_nonconformity_list)) * (1 - params.delta)))
        c = indirect_nonconformity_list[p - 1]
        # Perform robust conformal prediction.
        indirect_nonconformity_list = indirect_nonconformity_list[:-1]
        p_tilde = int(np.ceil(len(indirect_nonconformity_list) * (1 - delta_tilde)))
        c_tilde = indirect_nonconformity_list[p_tilde - 1]
        indirect_time = time.time() - indirect_start_time
        indirect_times.append(indirect_time)
        print("Calibration Time for the Indirect Method:", indirect_time, "seconds.")
        # Save the data for plotting.
        if experiment_num == 0:
            with open(f"experiment_results/{params.num_agents}-agent/STL/indirect_nonconformity_list.json", "w") as file:
                json.dump(indirect_nonconformity_list, file)
            with open(f"experiment_results/{params.num_agents}-agent/STL/old_indirect_nonconformity_list.json", "w") as file:
                json.dump(indirect_nonconformity_list_2023, file)
            with open(f"experiment_results/{params.num_agents}-agent/STL/c_indirect.txt", "w") as file:
                file.write(str(c))
            with open(f"experiment_results/{params.num_agents}-agent/STL/c_tilde_indirect.txt", "w") as file:
                file.write(str(c_tilde))
        # Generate the prediction regions.
        indirect_test_start_time = time.time()
        indirect_prediction_region_vanilla = dict()
        indirect_prediction_region_robust = dict()
        for tau in range(params.current_time + 1, params.T + 1):
            indirect_prediction_region_vanilla[tau] = c * alphas_indirect[tau]
            indirect_prediction_region_robust[tau] = c_tilde * alphas_indirect[tau]
        indirect_ground_robustnesses = []
        indirect_worst_robustnesses_vanilla = []
        indirect_worst_robustnesses_robust = []
        indirect_coverage_count_vanilla = 0
        indirect_coverage_count_robust = 0
        # Calculate the coverage for the indirect methods.
        testing_progress = 0
        for index in random_indices_test_trajectories:
            if (testing_progress + 1) % 10 == 0:
                print("Testing on test index:", testing_progress + 1, "out of", params.num_test_each_trial, "data points.")
            history = test_trajectories[index]
            indirect_ground_robustness = step_7_alpha_calculation_STL.evaluate_final_robustness(history)
            pred_history = test_predicted_trajectories[index]
            indirect_worst_robustness_vanilla = compute_worst_case_robustness_indirect(pred_history, indirect_prediction_region_vanilla)
            indirect_worst_robustness_robust = compute_worst_case_robustness_indirect(pred_history, indirect_prediction_region_robust)
            if indirect_ground_robustness >= indirect_worst_robustness_vanilla:
                indirect_coverage_count_vanilla += 1
            if indirect_ground_robustness >= indirect_worst_robustness_robust:
                indirect_coverage_count_robust += 1
            indirect_ground_robustnesses.append(indirect_ground_robustness)
            indirect_worst_robustnesses_vanilla.append(indirect_worst_robustness_vanilla)
            indirect_worst_robustnesses_robust.append(indirect_worst_robustness_robust)
            testing_progress += 1
        indirect_test_time = time.time() - indirect_test_start_time
        indirect_test_times.append(indirect_test_time)
        print("Test time for the indirect method", indirect_test_time, "seconds.")
        # Save the data.
        if experiment_num == 0:
            with open(f"experiment_results/{params.num_agents}-agent/STL/indirect_ground_robustnesses.json", "w") as file:
                json.dump(indirect_ground_robustnesses, file)
            with open(f"experiment_results/{params.num_agents}-agent/STL/indirect_worst_robustnesses_vanilla.json", "w") as file:
                json.dump(indirect_worst_robustnesses_vanilla, file)
            with open(f"experiment_results/{params.num_agents}-agent/STL/indirect_worst_robustnesses_robust.json", "w") as file:
                json.dump(indirect_worst_robustnesses_robust, file)
        indirect_vanilla_coverage = indirect_coverage_count_vanilla / params.num_test_each_trial
        indirect_robust_coverage = indirect_coverage_count_robust / params.num_test_each_trial
        indirect_vanilla_coverages.append(indirect_vanilla_coverage)
        indirect_robust_coverages.append(indirect_robust_coverage)
        print("Indirect Coverage Vanilla CP:", indirect_vanilla_coverage)
        print("Indirect Coverage Robust CP:", indirect_robust_coverage)
        print("=== Finished conducting the indirect method. ===")

        print("=== Conducting the Hybrid Method ===")
        # Time for the hybrid method.
        hybrid_start_time = time.time()
        hybrid_nonconformity_list = step_8_distribution_shift_computation_STL.calculate_hybrid_nonconformity(calib_trajectories, calib_predicted_trajectories, alphas_hybrid)
        hybrid_nonconformity_list.sort()
        # Perform vanilla conformal prediction.
        hybrid_nonconformity_list.append(float("inf"))
        # Find c.
        p = int(np.ceil((len(hybrid_nonconformity_list)) * (1 - params.delta)))
        c = hybrid_nonconformity_list[p - 1]
        # Perform robust conformal prediction.
        hybrid_nonconformity_list = hybrid_nonconformity_list[:-1]
        p_tilde = int(np.ceil(len(hybrid_nonconformity_list) * (1 - delta_tilde)))
        c_tilde = hybrid_nonconformity_list[p_tilde - 1]
        hybrid_time = time.time() - hybrid_start_time
        hybrid_times.append(hybrid_time)
        print("Calibration Time for the Hybrid Method:", hybrid_time, "seconds.")
        # Save the data.
        if experiment_num == 0:
            with open(f"experiment_results/{params.num_agents}-agent/STL/hybrid_nonconformity_list.json", "w") as file:
                json.dump(hybrid_nonconformity_list, file)
            with open(f"experiment_results/{params.num_agents}-agent/STL/c_hybrid.txt", "w") as file:
                file.write(str(c))
            with open(f"experiment_results/{params.num_agents}-agent/STL/c_tilde_hybrid.txt", "w") as file:
                file.write(str(c_tilde))
        hybrid_test_start_time = time.time()
        # Calculate the coverage for the hybrid methods.
        hybrid_ground_robustnesses = []
        hybrid_worst_robustnesses_vanilla = []
        hybrid_worst_robustnesses_robust = []
        hybrid_coverage_count_vanilla = 0
        hybrid_coverage_count_robust = 0
        testing_progress = 0
        for index in random_indices_test_trajectories:
            if (testing_progress + 1) % 10 == 0:
                print("Testing on test index:", testing_progress + 1, "out of", params.num_test_each_trial,
                      "data points.")
            history = test_trajectories[index]
            hybrid_ground_robustness = step_7_alpha_calculation_STL.evaluate_final_robustness(history)
            pred_history = test_predicted_trajectories[index]
            hybrid_worst_robustness_vanilla = compute_worst_case_robustness_hybrid(pred_history, alphas_hybrid, c)
            hybrid_worst_robustness_robust = compute_worst_case_robustness_hybrid(pred_history, alphas_hybrid, c_tilde)
            if hybrid_ground_robustness >= hybrid_worst_robustness_vanilla:
                hybrid_coverage_count_vanilla += 1
            if hybrid_ground_robustness >= hybrid_worst_robustness_robust:
                hybrid_coverage_count_robust += 1
            hybrid_ground_robustnesses.append(hybrid_ground_robustness)
            hybrid_worst_robustnesses_vanilla.append(hybrid_worst_robustness_vanilla)
            hybrid_worst_robustnesses_robust.append(hybrid_worst_robustness_robust)
            testing_progress += 1
        hybrid_test_time = time.time() - hybrid_test_start_time
        hybrid_test_times.append(hybrid_test_time)
        print("Test time for the hybrid method", hybrid_test_time, "seconds.")
        # Save the data.
        if experiment_num == 0:
            with open(f"experiment_results/{params.num_agents}-agent/STL/hybrid_ground_robustnesses.json", "w") as file:
                json.dump(hybrid_ground_robustnesses, file)
            with open(f"experiment_results/{params.num_agents}-agent/STL/hybrid_worst_robustnesses_vanilla.json", "w") as file:
                json.dump(hybrid_worst_robustnesses_vanilla, file)
            with open(f"experiment_results/{params.num_agents}-agent/STL/hybrid_worst_robustnesses_robust.json", "w") as file:
                json.dump(hybrid_worst_robustnesses_robust, file)
        hybrid_vanilla_coverage = hybrid_coverage_count_vanilla / params.num_test_each_trial
        hybrid_robust_coverage = hybrid_coverage_count_robust / params.num_test_each_trial
        hybrid_vanilla_coverages.append(hybrid_vanilla_coverage)
        hybrid_robust_coverages.append(hybrid_robust_coverage)
        print("Hybrid Coverage Vanilla CP:", hybrid_vanilla_coverage)
        print("Hybrid Coverage Robust CP:", hybrid_robust_coverage)
        print()

    print("=== Finished the main experiment procedure ===")
    print()

    # Finally, save the data for the coverages.
    with open(f"experiment_results/{params.num_agents}-agent/STL/direct_vanilla_coverages.json", "w") as file:
        json.dump(direct_vanilla_coverages, file)
    with open(f"experiment_results/{params.num_agents}-agent/STL/direct_robust_coverages.json", "w") as file:
        json.dump(direct_robust_coverages, file)
    with open(f"experiment_results/{params.num_agents}-agent/STL/indirect_vanilla_coverages.json", "w") as file:
        json.dump(indirect_vanilla_coverages, file)
    with open(f"experiment_results/{params.num_agents}-agent/STL/indirect_robust_coverages.json", "w") as file:
        json.dump(indirect_robust_coverages, file)
    with open(f"experiment_results/{params.num_agents}-agent/STL/hybrid_vanilla_coverages.json", "w") as file:
        json.dump(hybrid_vanilla_coverages, file)
    with open(f"experiment_results/{params.num_agents}-agent/STL/hybrid_robust_coverages.json", "w") as file:
        json.dump(hybrid_robust_coverages, file)

    # Save the timings.
    # Save timings for calculating C and c_tilde.
    with open(f"experiment_results/{params.num_agents}-agent/STL/direct_prediction_region_calculation_times.json",
              "w") as file:
        json.dump(direct_times, file)
    with open(f"experiment_results/{params.num_agents}-agent/STL/indirect_prediction_region_calculation_times.json",
              "w") as file:
        json.dump(indirect_times, file)
    with open(f"experiment_results/{params.num_agents}-agent/STL/hybrid_prediction_region_calculation_times.json",
              "w") as file:
        json.dump(hybrid_times, file)
    # Save timings for testing.
    with open(f"experiment_results/{params.num_agents}-agent/STL/direct_test_times.json", "w") as file:
        json.dump(direct_test_times, file)
    with open(f"experiment_results/{params.num_agents}-agent/STL/indirect_test_times.json", "w") as file:
        json.dump(indirect_test_times, file)
    with open(f"experiment_results/{params.num_agents}-agent/STL/hybrid_test_times.json", "w") as file:
        json.dump(hybrid_test_times, file)

    print("=== Experiment Ends ===")


if __name__ == '__main__':
    main()