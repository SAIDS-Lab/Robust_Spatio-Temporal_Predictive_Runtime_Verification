import params
import numpy as np
import step_0_data_processing
import step_1_data_analysis
import step_2_predictor_training
import step_4_distribution_shift_computation
import keras
import matplotlib.pyplot as plt
import json
import scipy
import time


def f_divergence(t):
    # We assume to use the total variation distance.
    return 0.5 * abs(t - 1)


def g(f, epsilon, beta, search_step = 0.0007):
    # Check input.
    if beta < 0 or beta > 1:
        raise Exception("Input to the function g is out of range.")
    # Perform a sampling-based line search.
    z = 0
    while z <= 1:
        value = beta * f(z / beta) + (1 - beta) * f((1 - z) / (1 - beta))
        if value <= epsilon:
            return z
        z += search_step
    raise Exception("No return from function g.")


def g_inverse(f, epsilon, tau, search_step = 0.0007):
    # Check input.
    if tau < 0 or tau > 1:
        raise Exception("Input to the function g_inverse is out of range.")
    beta = 1
    while beta >= 0:
        if beta != 1 and g(f, epsilon, beta) <= tau:
            return beta
        beta -= search_step
    raise Exception("No return from function g_inverse.")


def calculate_delta_n(delta, n, f, epsilon):
    inner = (1 + 1 / n) * g_inverse(f, epsilon, 1 - delta)
    return (1 - g(f, epsilon, inner))


def calculate_delta_tilde(delta_n, f, epsilon):
    answer = 1 - g_inverse(f, epsilon, 1 - delta_n)
    return answer


def find_optimal_communication(x_hat, radius):
    return params.terminal_height - (x_hat[2] + radius)


def find_optimal_ground(x_hat, radius):
    return (x_hat[2] - radius) - params.ground_height


def find_optimal_goal(x_hat, radius):
    return (x_hat[0] - radius) - params.goal_location


def closest_euclidean_distance_to_obstacles(x):
    centers, side_length = step_1_data_analysis.process_obstacles()
    closest_distance = float("inf")
    for center in centers:
        new_distance = max(abs(x[0] - center[0]), abs(x[1] - center[1])) - side_length / 2
        closest_distance = min(closest_distance, new_distance)
    return closest_distance


def find_optimal_obstacles(x_hat, radius):
    def cons_in_ball(x):
        return radius ** 2 - ((x[0] - x_hat[0]) ** 2 + (x[1] - x_hat[1]) ** 2 + (x[2] - x_hat[2]) ** 2)
    solution = scipy.optimize.minimize(closest_euclidean_distance_to_obstacles, x_hat, constraints = {'type': 'ineq', 'fun': cons_in_ball}).x
    return closest_euclidean_distance_to_obstacles(solution)


def calculate_success_reach_robustness(graph, predicate_matrix_obstacle, predicate_matrix_collision, predicate_matrix_goal, t1, t2):
    # Calculate the conjunction matrix.
    conjunction_matrix_obstacle_ground_avoidance = graph.calculate_conjunction_robustness(predicate_matrix_obstacle, predicate_matrix_collision)
    # Calculate the always matrix.
    always_conjunction_matrix_obstacle_ground_avoidance = graph.calculate_always_robustness(conjunction_matrix_obstacle_ground_avoidance, t1, t2)
    # Calculate the eventually matrix.
    eventually_matrix_goal_reaching = graph.calculate_eventually_robustness(predicate_matrix_goal, t1, t2)
    # Calculate the final matrix.
    final_robustness_matrix = graph.calculate_conjunction_robustness(always_conjunction_matrix_obstacle_ground_avoidance, eventually_matrix_goal_reaching)
    return final_robustness_matrix


def calculate_maintain_connection_terminal(graph, predicate_matrix_communication, t1, t2):
    # Calculate the somewhere matrix.
    somewhere_matrix_communication_to_terminal_check = graph.calculate_somewhere_robustness(predicate_matrix_communication, 0, params.communication_distance_threshold)
    # Calculate the always matrix.
    final_matrix = graph.calculate_always_robustness(somewhere_matrix_communication_to_terminal_check, t1, t2)
    return final_matrix


def calculate_final_robustness(graph, predicate_matrix_obstacle, predicate_matrix_collision, predicate_matrix_goal, predicate_matrix_communication, t1, t2):
    success_reach_matrix = calculate_success_reach_robustness(graph, predicate_matrix_obstacle, predicate_matrix_collision, predicate_matrix_goal, t1, t2)
    maintain_connection_matrix = calculate_maintain_connection_terminal(graph, predicate_matrix_communication, t1, t2)
    return graph.calculate_conjunction_robustness(success_reach_matrix, maintain_connection_matrix)


def compute_worst_case_robustness_indirect(graph, prediction_region):
    original_predicate_matrix_communication = graph.calculate_predicate_robustness(graph.communication_to_terminal_check)
    original_predicate_matrix_ground = graph.calculate_predicate_robustness(graph.ground_collision_avoidance_check)
    original_predicate_matrix_goal = graph.calculate_predicate_robustness(graph.goal_reaching)
    original_predicate_matrix_obstacles = graph.calculate_predicate_robustness(graph.closest_euclidean_distance_to_obstacles)
    worst_case_predicate_matrix_communication = dict()
    worst_case_predicate_matrix_ground = dict()
    worst_case_predicate_matrix_goal = dict()
    worst_case_predicate_matrix_obstacles = dict()
    for agent in range(params.num_agents):
        worst_case_predicate_matrix_communication[agent] = dict()
        worst_case_predicate_matrix_ground[agent] = dict()
        worst_case_predicate_matrix_goal[agent] = dict()
        worst_case_predicate_matrix_obstacles[agent] = dict()
        for tau in range(0, params.T + 1):
            if tau <= params.current_time:
                worst_case_predicate_matrix_communication[agent][tau] = original_predicate_matrix_communication[agent][tau]
                worst_case_predicate_matrix_ground[agent][tau] = original_predicate_matrix_ground[agent][tau]
                worst_case_predicate_matrix_goal[agent][tau] = original_predicate_matrix_goal[agent][tau]
                worst_case_predicate_matrix_obstacles[agent][tau] = original_predicate_matrix_obstacles[agent][tau]
            else:
                x_hat = graph.states[tau][agent]
                worst_case_predicate_matrix_communication[agent][tau] = find_optimal_communication(x_hat, prediction_region[tau][agent])
                worst_case_predicate_matrix_ground[agent][tau] = find_optimal_ground(x_hat, prediction_region[tau][agent])
                worst_case_predicate_matrix_goal[agent][tau] = find_optimal_goal(x_hat, prediction_region[tau][agent])
                worst_case_predicate_matrix_obstacles[agent][tau] = find_optimal_obstacles(x_hat, prediction_region[tau][agent])
    # Find worst case robustness for the final specification.
    worst_robustness = calculate_final_robustness(graph, worst_case_predicate_matrix_obstacles, worst_case_predicate_matrix_ground, worst_case_predicate_matrix_goal,
                               worst_case_predicate_matrix_communication, 0, params.T)[params.ego_agent][0]
    return worst_robustness


def compute_worst_case_robustness_hybrid(graph, alphas_hybrid, c_value):
    original_predicate_matrix_communication = graph.calculate_predicate_robustness(graph.communication_to_terminal_check)
    original_predicate_matrix_ground = graph.calculate_predicate_robustness(graph.ground_collision_avoidance_check)
    original_predicate_matrix_goal = graph.calculate_predicate_robustness(graph.goal_reaching)
    original_predicate_matrix_obstacles = graph.calculate_predicate_robustness(graph.closest_euclidean_distance_to_obstacles)
    worst_case_predicate_matrix_communication = dict()
    worst_case_predicate_matrix_ground = dict()
    worst_case_predicate_matrix_goal = dict()
    worst_case_predicate_matrix_obstacles = dict()
    for agent in range(params.num_agents):
        worst_case_predicate_matrix_communication[agent] = dict()
        worst_case_predicate_matrix_ground[agent] = dict()
        worst_case_predicate_matrix_goal[agent] = dict()
        worst_case_predicate_matrix_obstacles[agent] = dict()
        for tau in range(0, params.T + 1):
            if tau <= params.current_time:
                worst_case_predicate_matrix_communication[agent][tau] = original_predicate_matrix_communication[agent][tau]
                worst_case_predicate_matrix_ground[agent][tau] = original_predicate_matrix_ground[agent][tau]
                worst_case_predicate_matrix_goal[agent][tau] = original_predicate_matrix_goal[agent][tau]
                worst_case_predicate_matrix_obstacles[agent][tau] = original_predicate_matrix_obstacles[agent][tau]
            else:
                x_hat = graph.states[tau][agent]
                worst_case_predicate_matrix_communication[agent][tau] = params.terminal_height - x_hat[2] - c_value * alphas_hybrid["communication_to_terminal_check"][tau][agent]
                worst_case_predicate_matrix_ground[agent][tau] = x_hat[2] - params.ground_height - c_value * alphas_hybrid["ground_collision_avoidance_check"][tau][agent]
                worst_case_predicate_matrix_goal[agent][tau] = x_hat[0] - params.goal_location - c_value * alphas_hybrid["goal_reaching"][tau][agent]
                worst_case_predicate_matrix_obstacles[agent][tau] = closest_euclidean_distance_to_obstacles(x_hat) - c_value * alphas_hybrid["closest_euclidean_distance_to_obstacles"][tau][agent]
    # Find worst case robustness for the final specification.
    worst_robustness = calculate_final_robustness(graph, worst_case_predicate_matrix_obstacles, worst_case_predicate_matrix_ground, worst_case_predicate_matrix_goal,
                               worst_case_predicate_matrix_communication, 0, params.T)[params.ego_agent][0]
    return worst_robustness


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
    test_trajectories_all = step_0_data_processing.load_trajectories(shifted = True)
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
    predicted_calib_trajectories_all = step_2_predictor_training.generate_pred_trajectories(trained_lstm_models, calib_trajectories_all, lstm_norm)
    # Show one example of prediction.
    example_index = 210
    step_2_predictor_training.plot_predictions_2d(calib_trajectories_all[example_index], predicted_calib_trajectories_all[example_index], "Example Prediction from LSTM", "")
    print("=== Finished Producing Predicted Trajectories for the calibration data. ===")
    print()

    print("=== Produce Predicted Trajectories for the test data. ===")
    predicted_test_trajectories_all = step_2_predictor_training.generate_pred_trajectories(trained_lstm_models, test_trajectories_all, lstm_norm)
    # Show one example of prediction.
    example_index = 210
    step_2_predictor_training.plot_predictions_2d(test_trajectories_all[example_index], predicted_test_trajectories_all[example_index], "Example Prediction from LSTM", "")
    print("=== Finished Producing Predicted Trajectories for the test data. ===")
    print()

    print("=== Loading Epsilons ===")
    with open(f"epsilons/{params.num_agents}-agent/final_epsilon.txt", "r") as f:
        epsilon = float(f.read())
    print("=== Finished Loading Epsilons ===")
    print()

    print("=== Loading Alphas ===")
    with open(f'alphas/{params.num_agents}-agent/alphas_indirect.json', 'r') as f:
        alphas_indirect = json.load(f)
        new_alphas_indirect = dict()
        for tau in alphas_indirect:
            new_alphas_indirect[int(tau)] = dict()
            for agent in alphas_indirect[tau]:
                new_alphas_indirect[int(tau)][int(agent)] = alphas_indirect[tau][agent]
        alphas_indirect = new_alphas_indirect
    with open(f'alphas/{params.num_agents}-agent/alphas_hybrid.json', 'r') as f:
        alphas_hybrid = json.load(f)
        new_alphas_hybrid = dict()
        for predicate in alphas_hybrid:
            new_alphas_hybrid[predicate] = dict()
            for tau in alphas_hybrid[predicate]:
                new_alphas_hybrid[predicate][int(tau)] = dict()
                for agent in alphas_hybrid[predicate][tau]:
                    new_alphas_hybrid[predicate][int(tau)][int(agent)] = alphas_hybrid[predicate][tau][agent]
        alphas_hybrid = new_alphas_hybrid
    print("=== Finished Loading Alphas ===")
    print()

    print("=== Start the main experiment procedure ===")
    # Calculate delta_tilde.
    delta_n = calculate_delta_n(params.delta, params.num_calibration_each_trial, f_divergence, epsilon)
    delta_tilde = calculate_delta_tilde(delta_n, f_divergence, epsilon)
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
        # First, sample trajectories from the calibration trajectories all.
        random_indices_calib_trajectories = np.random.choice(list(calib_trajectories_all.keys()), params.num_calibration_each_trial, replace=False)
        calib_trajectories = dict()
        for index in random_indices_calib_trajectories:
            calib_trajectories[index] = calib_trajectories_all[index]
        calib_predicted_trajectories = dict()
        for index in random_indices_calib_trajectories:
            calib_predicted_trajectories[index] = predicted_calib_trajectories_all[index]
        # Sample trajectories from the test trajectories.
        random_indices_test_trajectories = np.random.choice(list(test_trajectories_all.keys()), params.num_test_each_trial, replace=False)
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
        direct_nonconformity_list = step_4_distribution_shift_computation.calculate_direct_nonconformity(calib_trajectories, calib_predicted_trajectories)
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
            with open(f"experiment_results/{params.num_agents}-agent/direct_nonconformity_list.json", "w") as file:
                json.dump(direct_nonconformity_list, file)
            with open(f"experiment_results/{params.num_agents}-agent/c_direct.txt", "w") as file:
                file.write(str(c))
            with open(f"experiment_results/{params.num_agents}-agent/c_tilde_direct.txt", "w") as file:
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
            graph = step_1_data_analysis.DynamicGraph(history)
            direct_ground_robustness = graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
            pred_history = test_predicted_trajectories[index]
            pred_graph = step_1_data_analysis.DynamicGraph(pred_history)
            pred_robustness = pred_graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
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
        # Save the data for the scatter plot of robustnesses.
        if experiment_num == 0:
            with open(f"experiment_results/{params.num_agents}-agent/direct_ground_robustnesses.json", "w") as file:
                json.dump(direct_ground_robustnesses, file)
            with open(f"experiment_results/{params.num_agents}-agent/direct_worst_robustnesses_vanilla.json", "w") as file:
                json.dump(direct_worst_robustnesses_vanilla, file)
            with open(f"experiment_results/{params.num_agents}-agent/direct_worst_robustnesses_robust.json", "w") as file:
                json.dump(direct_worst_robustnesses_robust, file)
        direct_vanilla_coverage = direct_coverage_count_vanilla / params.num_test_each_trial
        direct_robust_coverage = direct_coverage_count_robust / params.num_test_each_trial
        direct_vanilla_coverages.append(direct_vanilla_coverage)
        direct_robust_coverages.append(direct_robust_coverage)
        print("Direct Coverage Vanilla CP:", direct_vanilla_coverage)
        print("Direct Coverage Robust CP:", direct_robust_coverage)
        print("=== Finished conducting the direct method. ===")

        # Perform the indirect method.
        print("=== Conducting the indirect method. ===")
        # Time for the indirect method.
        indirect_start_time = time.time()
        indirect_nonconformity_list = step_4_distribution_shift_computation.calculate_indirect_nonconformity(calib_trajectories, calib_predicted_trajectories, alphas_indirect)
        indirect_nonconformity_list.sort()
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
        # Save the data for plotting the histogram over the nonconformity scores.
        if experiment_num == 0:
            with open(f"experiment_results/{params.num_agents}-agent/indirect_nonconformity_list.json", "w") as file:
                json.dump(indirect_nonconformity_list, file)
            with open(f"experiment_results/{params.num_agents}-agent/c_indirect.txt", "w") as file:
                file.write(str(c))
            with open(f"experiment_results/{params.num_agents}-agent/c_tilde_indirect.txt", "w") as file:
                file.write(str(c_tilde))
        # Generate the prediction regions.
        indirect_test_start_time = time.time()
        indirect_prediction_region_vanilla = dict()
        indirect_prediction_region_robust = dict()
        for tau in range(params.current_time + 1, params.T + 1):
            indirect_prediction_region_vanilla[tau] = dict()
            indirect_prediction_region_robust[tau] = dict()
            for agent in range(params.num_agents):
                indirect_prediction_region_vanilla[tau][agent] = c * alphas_indirect[tau][agent]
                indirect_prediction_region_robust[tau][agent] = c_tilde * alphas_indirect[tau][agent]
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
            graph = step_1_data_analysis.DynamicGraph(history)
            indirect_ground_robustness = graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
            pred_history = test_predicted_trajectories[index]
            pred_graph = step_1_data_analysis.DynamicGraph(pred_history)
            indirect_worst_robustness_vanilla = compute_worst_case_robustness_indirect(pred_graph, indirect_prediction_region_vanilla)
            indirect_worst_robustness_robust = compute_worst_case_robustness_indirect(pred_graph, indirect_prediction_region_robust)
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
        # Save the data for the scatter plot of robustnesses.
        if experiment_num == 0:
            with open(f"experiment_results/{params.num_agents}-agent/indirect_ground_robustnesses.json", "w") as file:
                json.dump(indirect_ground_robustnesses, file)
            with open(f"experiment_results/{params.num_agents}-agent/indirect_worst_robustnesses_vanilla.json", "w") as file:
                json.dump(indirect_worst_robustnesses_vanilla, file)
            with open(f"experiment_results/{params.num_agents}-agent/indirect_worst_robustnesses_robust.json", "w") as file:
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
        hybrid_nonconformity_list = step_4_distribution_shift_computation.calculate_hybrid_nonconformity(
            calib_trajectories, calib_predicted_trajectories, alphas_hybrid)
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
        # Save the data for the histogram over the nonconformity scores.
        if experiment_num == 0:
            with open(f"experiment_results/{params.num_agents}-agent/hybrid_nonconformity_list.json", "w") as file:
                json.dump(hybrid_nonconformity_list, file)
            with open(f"experiment_results/{params.num_agents}-agent/c_hybrid.txt", "w") as file:
                file.write(str(c))
            with open(f"experiment_results/{params.num_agents}-agent/c_tilde_hybrid.txt", "w") as file:
                file.write(str(c_tilde))
        hybrid_test_start_time = time.time()
        # Calculate the coverage for the hybrid methods.
        hybrid_ground_robustnesses = []
        hybrid_worst_robustnesses_vanilla = []
        hybrid_worst_robustnesses_robust = []
        hybrid_coverage_count_vanilla = 0
        hybrid_coverage_count_robust = 0
        # Calculate the coverage for the hybrid methods.
        testing_progress = 0
        for index in random_indices_test_trajectories:
            if (testing_progress + 1) % 10 == 0:
                print("Testing on test index:", testing_progress + 1, "out of", params.num_test_each_trial,
                      "data points.")
            history = test_trajectories[index]
            graph = step_1_data_analysis.DynamicGraph(history)
            hybrid_ground_robustness = graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
            pred_history = test_predicted_trajectories[index]
            pred_graph = step_1_data_analysis.DynamicGraph(pred_history)
            hybrid_worst_robustness_vanilla = compute_worst_case_robustness_hybrid(pred_graph, alphas_hybrid, c)
            hybrid_worst_robustness_robust = compute_worst_case_robustness_hybrid(pred_graph, alphas_hybrid, c_tilde)
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
        # Save the data for plotting the scatter plot of robustnesses.
        if experiment_num == 0:
            with open(f"experiment_results/{params.num_agents}-agent/hybrid_ground_robustnesses.json", "w") as file:
                json.dump(hybrid_ground_robustnesses, file)
            with open(f"experiment_results/{params.num_agents}-agent/hybrid_worst_robustnesses_vanilla.json", "w") as file:
                json.dump(hybrid_worst_robustnesses_vanilla, file)
            with open(f"experiment_results/{params.num_agents}-agent/hybrid_worst_robustnesses_robust.json", "w") as file:
                json.dump(hybrid_worst_robustnesses_robust, file)
        hybrid_vanilla_coverage = hybrid_coverage_count_vanilla / params.num_test_each_trial
        hybrid_robust_coverage = hybrid_coverage_count_robust / params.num_test_each_trial
        hybrid_vanilla_coverages.append(hybrid_vanilla_coverage)
        hybrid_robust_coverages.append(hybrid_robust_coverage)
        print("Hybrid Coverage Vanilla CP:", hybrid_vanilla_coverage)
        print("Hybrid Coverage Robust CP:", hybrid_robust_coverage)
        print("=== Finished Conducting the Hybrid Method ===")
        print()

    print("=== End of the main experiment procedure ===")
    print()

    # Finally, save the data for the coverages.
    with open(f"experiment_results/{params.num_agents}-agent/direct_vanilla_coverages.json", "w") as file:
        json.dump(direct_vanilla_coverages, file)
    with open(f"experiment_results/{params.num_agents}-agent/direct_robust_coverages.json", "w") as file:
        json.dump(direct_robust_coverages, file)
    with open(f"experiment_results/{params.num_agents}-agent/indirect_vanilla_coverages.json", "w") as file:
        json.dump(indirect_vanilla_coverages, file)
    with open(f"experiment_results/{params.num_agents}-agent/indirect_robust_coverages.json", "w") as file:
        json.dump(indirect_robust_coverages, file)
    with open(f"experiment_results/{params.num_agents}-agent/hybrid_vanilla_coverages.json", "w") as file:
        json.dump(hybrid_vanilla_coverages, file)
    with open(f"experiment_results/{params.num_agents}-agent/hybrid_robust_coverages.json", "w") as file:
        json.dump(hybrid_robust_coverages, file)

    # Save the timings.
    # Save timings for calculating C and c_tilde.
    with open(f"experiment_results/{params.num_agents}-agent/direct_prediction_region_calculation_times.json", "w") as file:
        json.dump(direct_times, file)
    with open(f"experiment_results/{params.num_agents}-agent/indirect_prediction_region_calculation_times.json", "w") as file:
        json.dump(indirect_times, file)
    with open(f"experiment_results/{params.num_agents}-agent/hybrid_prediction_region_calculation_times.json", "w") as file:
        json.dump(hybrid_times, file)
    # Save timings for testing.
    with open(f"experiment_results/{params.num_agents}-agent/direct_test_times.json", "w") as file:
        json.dump(direct_test_times, file)
    with open(f"experiment_results/{params.num_agents}-agent/indirect_test_times.json", "w") as file:
        json.dump(indirect_test_times, file)
    with open(f"experiment_results/{params.num_agents}-agent/hybrid_test_times.json", "w") as file:
        json.dump(hybrid_test_times, file)

    print("=== Experiment Ends ===")


if __name__ == '__main__':
    main()