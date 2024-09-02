import step_0_data_processing
import step_2_predictor_training
import step_7_alpha_calculation_STL
import step_4_distribution_shift_computation
import numpy as np
import params
import keras
import json


def calculate_direct_nonconformity(trajectories, predicted_trajectories, plot_example = False):
    robustnesses_ground = []
    indices = list(sorted(trajectories.keys()))
    for index in indices:
        history = trajectories[index]
        robustness = step_7_alpha_calculation_STL.evaluate_final_robustness(history)
        robustnesses_ground.append(robustness)
    robustnesses_predicted = []
    for index in indices:
        history = predicted_trajectories[index]
        robustness = step_7_alpha_calculation_STL.evaluate_final_robustness(history)
        robustnesses_predicted.append(robustness)
    # Calculate the direct nonconformity scores.
    nonconformity_scores = []
    for i in range(len(robustnesses_ground)):
        nonconformity_scores.append(robustnesses_predicted[i] - robustnesses_ground[i])

    # Plot one example of predicted_trajectories.
    if plot_example:
        print(trajectories[indices[20]])
        print(predicted_trajectories[indices[20]])
        print()
    return nonconformity_scores


def calculate_indirect_nonconformity(trajectories, predicted_trajectories, alphas_indirect):
    nonconformity_scores = []
    indices = list(sorted(trajectories.keys()))
    for index in indices:
        nonconformity_list_single = []
        for tau in range(params.current_time + 1, params.T + 1):
            x = trajectories[index][tau]
            x_hat = predicted_trajectories[index][tau]
            distance = ((x[0] - x_hat[0]) ** 2 + (x[1] - x_hat[1]) ** 2 + (x[2] - x_hat[2]) ** 2) ** (1 / 2)
            nonconformity_list_single.append(distance / (alphas_indirect[tau]))
        nonconformity_scores.append(max(nonconformity_list_single))
    return nonconformity_scores


def calculate_hybrid_nonconformity(trajectories, predicted_trajectories, alphas_hybrid):
    nonconformity_scores = []
    indices = list(sorted(trajectories.keys()))
    for index in indices:
        nonconformity_list_single = []
        for predicate in ["ground_collision", "obstacle_avoidance", "goal_reaching"]:
            for tau in range(params.current_time + 1, params.T + 1):
                if predicate == "ground_collision":
                    rho_hat = step_7_alpha_calculation_STL.evaluate_ground_collision_avoidance(predicted_trajectories[index], tau)
                    rho = step_7_alpha_calculation_STL.evaluate_ground_collision_avoidance(trajectories[index], tau)
                elif predicate == "obstacle_avoidance":
                    rho_hat = step_7_alpha_calculation_STL.evaluate_obstacle_avoidance(predicted_trajectories[index], tau)
                    rho = step_7_alpha_calculation_STL.evaluate_obstacle_avoidance(trajectories[index], tau)
                else:
                    rho_hat = step_7_alpha_calculation_STL.evaluate_goal_reaching(predicted_trajectories[index], tau)
                    rho = step_7_alpha_calculation_STL.evaluate_goal_reaching(trajectories[index], tau)
                nonconformity_list_single.append((rho_hat - rho) / (alphas_hybrid[predicate][tau]))
        nonconformity_scores.append(max(nonconformity_list_single))
    return nonconformity_scores


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
    with open(f"predictors/{params.num_agents}-agent/lstm/norm.txt", "r") as f:
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
    # Extract the ego agent's trajectories.
    trajectories_extracted = dict()
    for key in trajectories.keys():
        trajectories_extracted[key] = trajectories[key][params.ego_agent]
    trajectories = trajectories_extracted
    shifted_trajectories_extracted = dict()
    for key in shifted_trajectories.keys():
        shifted_trajectories_extracted[key] = shifted_trajectories[key][params.ego_agent]
    shifted_trajectories = shifted_trajectories_extracted
    predicted_trajectories_extracted = dict()
    for key in predicted_trajectories.keys():
        predicted_trajectories_extracted[key] = predicted_trajectories[key][params.ego_agent]
    predicted_trajectories = predicted_trajectories_extracted
    predicted_shifted_trajectories_extracted = dict()
    for key in predicted_shifted_trajectories.keys():
        predicted_shifted_trajectories_extracted[key] = predicted_shifted_trajectories[key][params.ego_agent]
    predicted_shifted_trajectories = predicted_shifted_trajectories_extracted
    # Compute the distribution shifts.
    print("=== Calculating the Distribution Shift for the Direct Method. ===")
    # Calculate the direct robustnesses for the nominal trajectories.
    direct_nonconformity_scores_nominal = calculate_direct_nonconformity(trajectories, predicted_trajectories)
    # Calculate the direct robustnesses for the shifted trajectories.
    direct_nonconformity_scores_shifted = calculate_direct_nonconformity(shifted_trajectories, predicted_shifted_trajectories)
    # Calculate the distribution shift.
    direct_distribution_shift = step_4_distribution_shift_computation.calculate_distribution_shift(direct_nonconformity_scores_nominal, direct_nonconformity_scores_shifted)
    print(f"Direct Distribution Shift: {direct_distribution_shift}")
    print("=== Finished Calculating the Distribution Shift for the Direct Method. ===")
    print()

    print("=== Calculating the Distribution Shift for the Indirect Method. ===")
    # Load the indirect alphas.
    with open(f"alphas/{params.num_agents}-agent/STL/alphas_indirect.json", "r") as f:
        alphas_indirect = json.load(f)
        new_alphas_indirect = dict()
        for tau in alphas_indirect:
            new_alphas_indirect[int(tau)] = alphas_indirect[tau]
        alphas_indirect = new_alphas_indirect
    # Calculate the indirect robsutnesses for the nominal trajectories.
    indirect_nonconformity_scores_nominal = calculate_indirect_nonconformity(trajectories, predicted_trajectories, alphas_indirect)
    # Calculate the indirect robustnesses for the shifted trajectories.
    indirect_nonconformity_scores_shifted = calculate_indirect_nonconformity(shifted_trajectories, predicted_shifted_trajectories, alphas_indirect)
    # Calculate the distribution shfit.
    indirect_distribution_shift = step_4_distribution_shift_computation.calculate_distribution_shift(indirect_nonconformity_scores_nominal, indirect_nonconformity_scores_shifted)
    print("Distribution Shift for the Indirect Method:", indirect_distribution_shift)
    print("=== Finished Calculating Distribution Shift for the Indirect Method. ===")
    print()

    print("=== Calculating the Distribution Shift for the Hybrid Method. ===")
    # Load hybrid alphas.
    with open(f"alphas/{params.num_agents}-agent/STL/alphas_hybrid.json", "r") as f:
        alphas_hybrid = json.load(f)
        new_alphas_hybrid = dict()
        for predicate in alphas_hybrid:
            new_alphas_hybrid[predicate] = dict()
            for tau in alphas_hybrid[predicate]:
                new_alphas_hybrid[predicate][int(tau)] = alphas_hybrid[predicate][tau]
        alphas_hybrid = new_alphas_hybrid
    # Calculate the hybrid robustnesses for the nominal trajectories.
    hybrid_nonconformity_scores_nominal = calculate_hybrid_nonconformity(trajectories, predicted_trajectories, alphas_hybrid)
    # Calculate the hybrid robustnesses for the shifted trajectories.
    hybrid_nonconformity_scores_shifted = calculate_hybrid_nonconformity(shifted_trajectories, predicted_shifted_trajectories, alphas_hybrid)
    # Calculate the distribution shift.
    hybrid_distribution_shift = step_4_distribution_shift_computation.calculate_distribution_shift(hybrid_nonconformity_scores_nominal, hybrid_nonconformity_scores_shifted)
    print("Distribution Shift for the Hybrid Method:", hybrid_distribution_shift)
    print("=== Finished Calculating Distribution Shift for the Hybrid Method. ===")
    print()

    print("=== Saving the results. ===")
    with open(f"epsilons/{params.num_agents}-agent/STL/direct_epsilon.txt", "w") as f:
        f.write(str(direct_distribution_shift))
    with open(f"epsilons/{params.num_agents}-agent/STL/indirect_epsilon.txt", "w") as f:
        f.write(str(indirect_distribution_shift))
    with open(f"epsilons/{params.num_agents}-agent/STL/hybrid_epsilon.txt", "w") as f:
        f.write(str(hybrid_distribution_shift))
    with open(f"epsilons/{params.num_agents}-agent/STL/final_epsilon.txt", "w") as f:
        f.write(str(max(direct_distribution_shift, indirect_distribution_shift, hybrid_distribution_shift)))
    print("=== Finished Saving the Results. ===")


if __name__ == '__main__':
    main()