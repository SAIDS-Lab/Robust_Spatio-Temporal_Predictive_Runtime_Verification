import params
import step_0_data_processing
import step_1_data_analysis
import numpy as np
import keras
from scipy.stats import gaussian_kde
import json
import step_2_predictor_training


def closest_euclidean_distance_to_obstacles(x):
    centers, side_length = step_1_data_analysis. process_obstacles()
    closest_distance = float("inf")
    for center in centers:
        new_distance = max(abs(x[0] - center[0]),
                           abs(x[1] - center[1])) - side_length / 2
        closest_distance = min(closest_distance, new_distance)
    return closest_distance


def calculate_direct_nonconformity(trajectories, predicted_trajectories, plot_example = False):
    # First, calculate the robustnesses over the ground truth trajectories.
    robustnesses_ground = []
    indices = list(sorted(trajectories.keys()))
    for index in indices:
        history = trajectories[index]
        graph = step_1_data_analysis.DynamicGraph(history)
        robustness = graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
        robustnesses_ground.append(robustness)
    # Calculate robustnesses over the predicted trajectories.
    robustnesses_pred = []
    for index in indices:
        history = predicted_trajectories[index]
        graph = step_1_data_analysis.DynamicGraph(history)
        robustness = graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
        robustnesses_pred.append(robustness)
    # Calculate the direct nonconformity scores.
    nonconformity_scores = []
    for i in range(len(robustnesses_ground)):
        nonconformity_scores.append(robustnesses_pred[i] - robustnesses_ground[i])

    # Plot one example of predicted trajectories.
    if plot_example:
        step_2_predictor_training.plot_predictions_2d(trajectories[indices[20]], predicted_trajectories[indices[20]], "Example", "")
    return nonconformity_scores


def calculate_indirect_nonconformity(trajectories, predicted_trajectories, alphas_indirect):
    nonconformity_scores = []
    indices = list(sorted(trajectories.keys()))
    for index in indices:
        nonconformity_list_single = []
        for tau in range(params.current_time + 1, params.T + 1):
            for agent in range(params.num_agents):
                x = trajectories[index][agent][tau]
                x_hat = predicted_trajectories[index][agent][tau]
                distance = ((x[0] - x_hat[0]) ** 2 + (x[1] - x_hat[1]) ** 2 + (x[2] - x_hat[2]) ** 2) ** (1 / 2)
                nonconformity_list_single.append(distance / alphas_indirect[tau][agent])
        nonconformity_scores.append(max(nonconformity_list_single))
    return nonconformity_scores


def calculate_hybrid_nonconformity(trajectories, predicted_trajectories, alphas_hybrid):
    nonconformity_scores = []
    indices = list(sorted(trajectories.keys()))
    for index in indices:
        nonconformity_list_single = []
        for predicate in ["communication_to_terminal_check", "ground_collision_avoidance_check", "goal_reaching", "closest_euclidean_distance_to_obstacles"]:
            for tau in range(params.current_time + 1, params.T + 1):
                for agent in range(params.num_agents):
                    if predicate == "communication_to_terminal_check":
                        rho_hat = params.terminal_height - predicted_trajectories[index][agent][tau][2]
                        rho = params.terminal_height - trajectories[index][agent][tau][2]
                    elif predicate == "ground_collision_avoidance_check":
                        rho_hat = predicted_trajectories[index][agent][tau][2] - params.ground_height
                        rho = trajectories[index][agent][tau][2] - params.ground_height
                    elif predicate == "goal_reaching":
                        rho_hat = predicted_trajectories[index][agent][tau][0] - params.goal_location
                        rho = trajectories[index][agent][tau][0] - params.goal_location
                    else:
                        rho_hat = closest_euclidean_distance_to_obstacles(predicted_trajectories[index][agent][tau])
                        rho = closest_euclidean_distance_to_obstacles(trajectories[index][agent][tau])
                    nonconformity_list_single.append((rho_hat - rho) / alphas_hybrid[predicate][tau][agent])
        nonconformity_scores.append(max(nonconformity_list_single))
    return nonconformity_scores


def calculate_distribution_shift(d_0_nonconformity_list, d_nonconformity_list):
    lower_bound = np.min(np.concatenate((d_0_nonconformity_list, d_nonconformity_list)))
    upper_bound = np.max(np.concatenate((d_0_nonconformity_list, d_nonconformity_list)))
    kde_d_0 = gaussian_kde(d_0_nonconformity_list)
    kde_d = gaussian_kde(d_nonconformity_list)
    # Now, compute the total variation.
    step_size = (upper_bound - lower_bound) / params.kde_calculation_bin_num
    new_score_list = np.arange(lower_bound, upper_bound, step_size)
    d_0_pdf = kde_d_0.evaluate(new_score_list)
    d_pdf = kde_d.evaluate(new_score_list)
    divergence = 0
    for i in range(len(new_score_list) - 1):
        y_front = 0.5 * abs(d_0_pdf[i] - d_pdf[i])
        y_back = 0.5 * abs(d_0_pdf[i + 1] - d_pdf[i + 1])
        divergence += ((y_front + y_back) * step_size / 2)
    return divergence


def main():
    np.random.seed(params.my_seed)
    # Load the trajectories.
    trajectories = step_0_data_processing.load_trajectories()
    # Load only the calibration trajectories.
    calib_trajectories = dict()
    for i in range(params.num_train + 1, params.num_histories + 1):
        calib_trajectories[i] = trajectories[i]
    trajectories = calib_trajectories
    shifted_trajectories = step_0_data_processing.load_trajectories(shifted = True)
    # Load the norm.
    with open(f'predictors/{params.num_agents}-agent/lstm/norm.txt', 'r') as f:
        lstm_norm = float(f.read())
    # Perform predictions.
    # Load the LSTM models.
    print("=== Loading the LSTM Model ===")
    trained_lstm_models = dict()
    for a in range(params.num_agents):
        trained_lstm_models[a] = dict()
        for s in range(3):
            trained_lstm_models[a][s] = keras.models.load_model(
                f"predictors/{params.num_agents}-agent/lstm/lstm_model_agent_{a}_state_{s}.keras")
    print("=== Finished Loading the LSTM Model ===")
    predicted_trajectories = step_2_predictor_training.generate_pred_trajectories(trained_lstm_models, trajectories, lstm_norm)
    predicted_shifted_trajectories = step_2_predictor_training.generate_pred_trajectories(trained_lstm_models, shifted_trajectories, lstm_norm)
    print("=== Calculating Distribution Shift for the direct method. ===")
    # Calculate the direct robustnesses for the nominal trajectories.
    direct_nonconformity_scores_nominal = calculate_direct_nonconformity(trajectories, predicted_trajectories)
    # Calculate the direct robustnesses for the shifted trajectories.
    direct_nonconformity_scores_shifted = calculate_direct_nonconformity(shifted_trajectories, predicted_shifted_trajectories)
    # Calculate the distribution shift.
    direct_distribution_shift = calculate_distribution_shift(direct_nonconformity_scores_nominal, direct_nonconformity_scores_shifted)
    print("Distribution Shift for the Direct Method:", direct_distribution_shift)
    print("=== Finished Calculating Distribution Shift for the direct method. ===")
    print()

    print("=== Calculating Distribution Shift for the indirect method. ===")
    # Load indirect alphas.
    with open(f'alphas/{params.num_agents}-agent/alphas_indirect.json', 'r') as f:
        alphas_indirect = json.load(f)
        new_alphas_indirect = dict()
        for tau in alphas_indirect:
            new_alphas_indirect[int(tau)] = dict()
            for agent in alphas_indirect[tau]:
                new_alphas_indirect[int(tau)][int(agent)] = alphas_indirect[tau][agent]
        alphas_indirect = new_alphas_indirect
    # Calculate the indirect robustnesses for the nominal trajectories.
    indirect_nonconformity_scores_nominal = calculate_indirect_nonconformity(trajectories, predicted_trajectories, alphas_indirect)
    # Calculate the indirect robustnesses for the shifted trajectories.
    indirect_nonconformity_scores_shifted = calculate_indirect_nonconformity(shifted_trajectories, predicted_shifted_trajectories, alphas_indirect)
    # Calculate the distribution shift.
    indirect_distribution_shift = calculate_distribution_shift(indirect_nonconformity_scores_nominal, indirect_nonconformity_scores_shifted)
    print("Distribution Shift for the Indirect Method:", indirect_distribution_shift)
    print("=== Finished Calculating Distribution Shift for the indirect method. ===")
    print()

    print("=== Calculating Distribution Shift for the hybrid method. ===")
    # Load hybrid alphas.
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
    # Calculate the hybrid robustnesses for the nominal trajectories.
    hybrid_nonconformity_scores_nominal = calculate_hybrid_nonconformity(trajectories, predicted_trajectories, alphas_hybrid)
    hybrid_nonconformity_scores_shifted = calculate_hybrid_nonconformity(shifted_trajectories, predicted_shifted_trajectories, alphas_hybrid)
    # Calculate the distribution shift.
    hybrid_distribution_shift = calculate_distribution_shift(hybrid_nonconformity_scores_nominal, hybrid_nonconformity_scores_shifted)
    print("Distribution Shift for the Hybrid Method:", hybrid_distribution_shift)
    print("=== Finished Calculating Distribution Shift for the hybrid method. ===")
    print()

    print("=== Saving the results ===")
    with open(f'epsilons/{params.num_agents}-agent/direct_epsilon.txt', 'w') as f:
        f.write(str(direct_distribution_shift))
    with open(f'epsilons/{params.num_agents}-agent/indirect_epsilon.txt', 'w') as f:
        f.write(str(indirect_distribution_shift))
    with open(f'epsilons/{params.num_agents}-agent/hybrid_epsilon.txt', 'w') as f:
        f.write(str(hybrid_distribution_shift))
    with open(f'epsilons/{params.num_agents}-agent/final_epsilon.txt', 'w') as f:
        f.write(str(max(direct_distribution_shift, indirect_distribution_shift, hybrid_distribution_shift)))
    print("=== Saving results finished. ===")


if __name__ == '__main__':
    main()