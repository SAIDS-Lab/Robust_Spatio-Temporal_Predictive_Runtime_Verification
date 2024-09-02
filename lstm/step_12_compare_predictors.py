import numpy as np
import step_0_data_processing
import step_1_data_analysis
import step_2_predictor_training
import params
import keras
import cnn.step_1_data_analysis
import cnn.step_2_predictor_training
import cnn.step_4_distribution_shift_computation
import transformer.step_1_data_analysis
import transformer.step_2_predictor_training
import transformer.step_4_distribution_shift_computation
import step_4_distribution_shift_computation
import step_5_experiments
import matplotlib.pyplot as plt
import json
import seaborn as sns

num_bins = 30
def main():
    # Load calibration trajectories.
    np.random.seed(params.my_seed)
    print("=== Loading the Calibration Trajectories ===")
    trajectories = step_0_data_processing.load_trajectories()
    calib_trajectories_all = dict()
    for i in range(params.num_train + 1, params.num_histories + 1):
        calib_trajectories_all[i] = trajectories[i]
    # Extract a calib sample size of trajectories.
    random_indices_calib_trajectories = np.random.choice(list(calib_trajectories_all.keys()), params.num_calibration_each_trial, replace=False)
    calib_trajectories = dict()
    for index in random_indices_calib_trajectories:
        calib_trajectories[index] = calib_trajectories_all[index]
    print("=== Finished Loading the Calibration Trajectories ===")

    print("=== Loading the Test Trajectories ===")
    test_trajectories_all = step_0_data_processing.load_trajectories(shifted = True)
    random_indices_test_trajectories = np.random.choice(list(test_trajectories_all.keys()), params.num_test_each_trial,
                                                        replace=False)
    test_trajectories = dict()
    for index in random_indices_test_trajectories:
        test_trajectories[index] = test_trajectories_all[index]
    print("=== Finished Loading the Test Trajectories ===")

    # Load predictors.
    print("=== Loading the LSTM Model ===")
    trained_lstm_models = dict()
    for a in range(params.num_agents):
        trained_lstm_models[a] = dict()
        for s in range(3):
            trained_lstm_models[a][s] = keras.models.load_model(
                f"predictors/{params.num_agents}-agent/lstm/lstm_model_agent_{a}_state_{s}.keras")
    print("=== Finished Loading the LSTM Model ===")
    print()

    print("=== Loading the CNN Model ===")
    trained_cnn_models = dict()
    for a in range(params.num_agents):
        trained_cnn_models[a] = dict()
        for s in range(3):
            trained_cnn_models[a][s] = keras.models.load_model(
                f"../cnn/predictors/{params.num_agents}-agent/cnn/cnn_model_agent_{a}_state_{s}.keras")
    print("=== Finished Loading the CNN Model ===")

    print("=== Loading the Transformer Model ===")
    trained_transformer_models = dict()
    for a in range(params.num_agents):
        trained_transformer_models[a] = dict()
        for s in range(3):
            trained_transformer_models[a][s] = keras.models.load_model(
                f"../transformer/predictors/{params.num_agents}-agent/transformer/transformer_model_agent_{a}_state_{s}.keras")
    print("=== Finished Loading the Transformer Model ===")

    # Load norms.
    print("=== Load the Norm for the LSTM Model ===")
    with open(f'predictors/{params.num_agents}-agent/lstm/norm.txt', 'r') as f:
        lstm_norm = float(f.read())
    print("=== Finished Loading the Norm for the LSTM Model ===")
    print()

    print("=== Load the Norm for the CNN Model ===")
    with open(f'../cnn/predictors/{params.num_agents}-agent/cnn/norm.txt', 'r') as f:
        cnn_norm = float(f.read())
    print("=== Finished Loading the Norm for the CNN Model ===")
    print()

    print("=== Load the Norm for the Transformer Model ===")
    with open(f'../transformer/predictors/{params.num_agents}-agent/transformer/norm.txt', 'r') as f:
        transformer_norm = float(f.read())
    print("=== Finished Loading the Norm for the Transformer Model ===")


    # Make predictions using all three predictors.
    print("=== Making Predictions ===")
    predicted_calib_trajectories_lstm = step_2_predictor_training.generate_pred_trajectories(trained_lstm_models, calib_trajectories, lstm_norm)
    predicted_calib_trajectories_cnn = cnn.step_2_predictor_training.generate_pred_trajectories(trained_cnn_models, calib_trajectories, cnn_norm)
    predicted_calib_trajectories_transformer = transformer.step_2_predictor_training.generate_pred_trajectories(trained_transformer_models, calib_trajectories, transformer_norm)
    predicted_test_trajectories_lstm = step_2_predictor_training.generate_pred_trajectories(trained_lstm_models, test_trajectories, lstm_norm)
    predicted_test_trajectories_cnn = cnn.step_2_predictor_training.generate_pred_trajectories(trained_cnn_models, test_trajectories, cnn_norm)
    predicted_test_trajectories_transformer = transformer.step_2_predictor_training.generate_pred_trajectories(trained_transformer_models, test_trajectories, transformer_norm)
    print("=== Finished Making Predictions ===")

    # Calculate the errors.
    experiment_agent = 0
    experiment_tau = 120
    experiment_pi = "goal_reaching"
    print("=== Calculating the Errors ===")
    calib_prediction_errors_lstm = dict()
    calib_prediction_errors_cnn = dict()
    calib_prediction_errors_transformer = dict()
    calib_prediction_errors_variance_lstm = dict()
    calib_prediction_errors_variance_cnn = dict()
    calib_prediction_errors_variance_transformer = dict()
    calib_indirect_nonconformity_2023_lstm = []
    calib_indirect_nonconformity_2023_cnn = []
    calib_indirect_nonconformity_2023_transformer = []
    calib_hybrid_nonconformity_2023_lstm = []
    calib_hybrid_nonconformity_2023_cnn = []
    calib_hybrid_nonconformity_2023_transformer = []
    for tau in range(params.current_time + 1, params.T + 1):
        lstm_error_list = []
        cnn_error_list = []
        transformer_error_list = []
        for agent in range(params.num_agents):
            for index in random_indices_calib_trajectories:
                x = calib_trajectories[index][agent][tau]
                x_hat_lstm = predicted_calib_trajectories_lstm[index][agent][tau]
                x_hat_cnn = predicted_calib_trajectories_cnn[index][agent][tau]
                x_hat_transformer = predicted_calib_trajectories_transformer[index][agent][tau]
                lstm_error = ((x[0] - x_hat_lstm[0]) ** 2 + (x[1] - x_hat_lstm[1]) ** 2 + (x[2] - x_hat_lstm[2]) ** 2) ** (1 / 2)
                cnn_error = ((x[0] - x_hat_cnn[0]) ** 2 + (x[1] - x_hat_cnn[1]) ** 2 + (x[2] - x_hat_cnn[2]) ** 2) ** (1 / 2)
                transformer_error = ((x[0] - x_hat_transformer[0]) ** 2 + (x[1] - x_hat_transformer[1]) ** 2 + (x[2] - x_hat_transformer[2]) ** 2) ** (1 / 2)
                lstm_error_list.append(lstm_error)
                cnn_error_list.append(cnn_error)
                transformer_error_list.append(transformer_error)
                if agent == experiment_agent and tau == experiment_tau:
                    calib_indirect_nonconformity_2023_lstm.append(lstm_error)
                    calib_indirect_nonconformity_2023_cnn.append(cnn_error)
                    calib_indirect_nonconformity_2023_transformer.append(transformer_error)
        calib_prediction_errors_lstm[tau] = sum(lstm_error_list) / (params.num_agents * params.num_calibration_each_trial)
        calib_prediction_errors_cnn[tau] = sum(cnn_error_list) / (params.num_agents * params.num_calibration_each_trial)
        calib_prediction_errors_transformer[tau] = sum(transformer_error_list) / (params.num_agents * params.num_calibration_each_trial)
        calib_prediction_errors_variance_lstm[tau] = np.var(lstm_error_list)
        calib_prediction_errors_variance_cnn[tau] = np.var(cnn_error_list)
        calib_prediction_errors_variance_transformer[tau] = np.var(transformer_error_list)

    for index in random_indices_calib_trajectories:
        x = calib_trajectories[index][experiment_agent][experiment_tau]
        x_hat_lstm = predicted_calib_trajectories_lstm[index][experiment_agent][experiment_tau]
        x_hat_cnn = predicted_calib_trajectories_cnn[index][experiment_agent][experiment_tau]
        x_hat_transformer = predicted_calib_trajectories_transformer[index][experiment_agent][experiment_tau]
        calib_hybrid_nonconformity_2023_lstm.append((x_hat_lstm[0] - params.goal_location) - (x[0] - params.goal_location))
        calib_hybrid_nonconformity_2023_cnn.append((x_hat_cnn[0] - params.goal_location) - (x[0] - params.goal_location))
        calib_hybrid_nonconformity_2023_transformer.append((x_hat_transformer[0] - params.goal_location) - (x[0] - params.goal_location))

    test_prediction_errors_lstm = dict()
    test_prediction_errors_cnn = dict()
    test_prediction_errors_transformer = dict()
    test_prediction_errors_variance_lstm = dict()
    test_prediction_errors_variance_cnn = dict()
    test_prediction_errors_variance_transformer = dict()
    for tau in range(params.current_time + 1, params.T + 1):
        lstm_error_list = []
        cnn_error_list = []
        transformer_error_list = []
        for agent in range(params.num_agents):
            for index in random_indices_test_trajectories:
                x = test_trajectories[index][agent][tau]
                x_hat_lstm = predicted_test_trajectories_lstm[index][agent][tau]
                x_hat_cnn = predicted_test_trajectories_cnn[index][agent][tau]
                x_hat_transformer = predicted_test_trajectories_transformer[index][agent][tau]
                lstm_error = ((x[0] - x_hat_lstm[0]) ** 2 + (x[1] - x_hat_lstm[1]) ** 2 + (x[2] - x_hat_lstm[2]) ** 2) ** (1 / 2)
                cnn_error = ((x[0] - x_hat_cnn[0]) ** 2 + (x[1] - x_hat_cnn[1]) ** 2 + (x[2] - x_hat_cnn[2]) ** 2) ** (1 / 2)
                transformer_error = ((x[0] - x_hat_transformer[0]) ** 2 + (x[1] - x_hat_transformer[1]) ** 2 + (x[2] - x_hat_transformer[2]) ** 2) ** (1 / 2)
                lstm_error_list.append(lstm_error)
                cnn_error_list.append(cnn_error)
                transformer_error_list.append(transformer_error)
        test_prediction_errors_lstm[tau] = sum(lstm_error_list) / (params.num_agents * params.num_test_each_trial)
        test_prediction_errors_cnn[tau] = sum(cnn_error_list) / (params.num_agents * params.num_test_each_trial)
        test_prediction_errors_transformer[tau] = sum(transformer_error_list) / (params.num_agents * params.num_test_each_trial)
        test_prediction_errors_variance_lstm[tau] = np.var(lstm_error_list)
        test_prediction_errors_variance_cnn[tau] = np.var(cnn_error_list)
        test_prediction_errors_variance_transformer[tau] = np.var(transformer_error_list)
    print("=== Finished Calculating the Errors ===")

    # plot the errors.
    print("=== Plotting the Errors ===")
    plt.scatter([tau for tau in range(params.current_time + 1, params.T + 1)], [calib_prediction_errors_lstm[tau] for tau in range(params.current_time + 1, params.T + 1)], label="LSTM: Calibration Set", color="r")
    plt.scatter([tau for tau in range(params.current_time + 1, params.T + 1)], [calib_prediction_errors_cnn[tau] for tau in range(params.current_time + 1, params.T + 1)], label="CNN: Calibration Set", color="g")
    plt.scatter([tau for tau in range(params.current_time + 1, params.T + 1)], [calib_prediction_errors_transformer[tau] for tau in range(params.current_time + 1, params.T + 1)], label="Transformer: Calibration Set", color="b")
    plt.scatter([tau for tau in range(params.current_time + 1, params.T + 1)],
                [test_prediction_errors_lstm[tau] for tau in range(params.current_time + 1, params.T + 1)],
                label="LSTM: Test Set", color="k")
    plt.scatter([tau for tau in range(params.current_time + 1, params.T + 1)],
                [test_prediction_errors_cnn[tau] for tau in range(params.current_time + 1, params.T + 1)], label="CNN: Test Set",
                color="m")
    plt.scatter([tau for tau in range(params.current_time + 1, params.T + 1)],
                [test_prediction_errors_transformer[tau] for tau in range(params.current_time + 1, params.T + 1)], label="Transformer: Test Set",  color="c")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Time", fontsize=params.font_size)
    plt.ylabel("Prediction Error Mean", fontsize=params.font_size)
    plt.legend(fontsize=params.legend_size)
    plt.tight_layout()
    plt.savefig("comparison_plots/prediction_errors_mean.pdf")
    plt.show()

    plt.scatter([tau for tau in range(params.current_time + 1, params.T + 1)], [calib_prediction_errors_variance_lstm[tau] for tau in range(params.current_time + 1, params.T + 1)], label="LSTM: Calibration Set", color="r")
    plt.scatter([tau for tau in range(params.current_time + 1, params.T + 1)], [calib_prediction_errors_variance_cnn[tau] for tau in range(params.current_time + 1, params.T + 1)], label="CNN: Calibration Set", color="g")
    plt.scatter([tau for tau in range(params.current_time + 1, params.T + 1)], [calib_prediction_errors_variance_transformer[tau] for tau in range(params.current_time + 1, params.T + 1)], label="Transformer: Calibration Set", color="b")
    plt.scatter([tau for tau in range(params.current_time + 1, params.T + 1)], [test_prediction_errors_variance_lstm[tau] for tau in range(params.current_time + 1, params.T + 1)], label="LSTM: Test Set", color="k")
    plt.scatter([tau for tau in range(params.current_time + 1, params.T + 1)], [test_prediction_errors_variance_cnn[tau] for tau in range(params.current_time + 1, params.T + 1)], label="CNN: Test Set", color="m")
    plt.scatter([tau for tau in range(params.current_time + 1, params.T + 1)], [test_prediction_errors_variance_transformer[tau] for tau in range(params.current_time + 1, params.T + 1)], label="Transformer: Test Set", color="c")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Time", fontsize=params.font_size)
    plt.ylabel("Prediction Error Variance", fontsize=params.font_size)
    plt.legend(fontsize=params.legend_size)
    plt.tight_layout()
    plt.savefig("comparison_plots/prediction_errors_variance.pdf")
    plt.show()
    print("=== Finished Plotting the Errors ===")

    print("=== Loading Epsilons ===")
    with open(f"epsilons/{params.num_agents}-agent/final_epsilon.txt", "r") as f:
        lstm_epsilon = float(f.read())
    with open(f"../cnn/epsilons/{params.num_agents}-agent/final_epsilon.txt", "r") as f:
        cnn_epsilon = float(f.read())
    with open(f"../transformer/epsilons/{params.num_agents}-agent/final_epsilon.txt", "r") as f:
        transformer_epsilon = float(f.read())
    epsilon = max(lstm_epsilon, cnn_epsilon, transformer_epsilon)
    print("Using Epsilon:", epsilon)
    print("=== Finished Loading Epsilons ===")
    print()

    print("=== Loading Alphas ===")
    with open(f'alphas/{params.num_agents}-agent/alphas_indirect.json', 'r') as f:
        lstm_alphas_indirect = json.load(f)
        lstm_new_alphas_indirect = dict()
        for tau in lstm_alphas_indirect:
            lstm_new_alphas_indirect[int(tau)] = dict()
            for agent in lstm_alphas_indirect[tau]:
                lstm_new_alphas_indirect[int(tau)][int(agent)] = lstm_alphas_indirect[tau][agent]
        lstm_alphas_indirect = lstm_new_alphas_indirect
    with open(f'alphas/{params.num_agents}-agent/alphas_hybrid.json', 'r') as f:
        lstm_alphas_hybrid = json.load(f)
        new_lstm_alphas_hybrid = dict()
        for predicate in lstm_alphas_hybrid:
            new_lstm_alphas_hybrid[predicate] = dict()
            for tau in lstm_alphas_hybrid[predicate]:
                new_lstm_alphas_hybrid[predicate][int(tau)] = dict()
                for agent in lstm_alphas_hybrid[predicate][tau]:
                    new_lstm_alphas_hybrid[predicate][int(tau)][int(agent)] = lstm_alphas_hybrid[predicate][tau][agent]
        lstm_alphas_hybrid = new_lstm_alphas_hybrid
    with open(f'../cnn/alphas/{params.num_agents}-agent/alphas_indirect.json', 'r') as f:
        cnn_alphas_indirect = json.load(f)
        cnn_new_alphas_indirect = dict()
        for tau in cnn_alphas_indirect:
            cnn_new_alphas_indirect[int(tau)] = dict()
            for agent in cnn_alphas_indirect[tau]:
                cnn_new_alphas_indirect[int(tau)][int(agent)] = cnn_alphas_indirect[tau][agent]
        cnn_alphas_indirect = cnn_new_alphas_indirect
    with open(f'../cnn/alphas/{params.num_agents}-agent/alphas_hybrid.json', 'r') as f:
        cnn_alphas_hybrid = json.load(f)
        new_cnn_alphas_hybrid = dict()
        for predicate in cnn_alphas_hybrid:
            new_cnn_alphas_hybrid[predicate] = dict()
            for tau in cnn_alphas_hybrid[predicate]:
                new_cnn_alphas_hybrid[predicate][int(tau)] = dict()
                for agent in cnn_alphas_hybrid[predicate][tau]:
                    new_cnn_alphas_hybrid[predicate][int(tau)][int(agent)] = cnn_alphas_hybrid[predicate][tau][agent]
        cnn_alphas_hybrid = new_cnn_alphas_hybrid
    with open(f'../transformer/alphas/{params.num_agents}-agent/alphas_indirect.json', 'r') as f:
        transformer_alphas_indirect = json.load(f)
        transformer_new_alphas_indirect = dict()
        for tau in transformer_alphas_indirect:
            transformer_new_alphas_indirect[int(tau)] = dict()
            for agent in transformer_alphas_indirect[tau]:
                transformer_new_alphas_indirect[int(tau)][int(agent)] = transformer_alphas_indirect[tau][agent]
        transformer_alphas_indirect = transformer_new_alphas_indirect
    with open(f'../transformer/alphas/{params.num_agents}-agent/alphas_hybrid.json', 'r') as f:
        transformer_alphas_hybrid = json.load(f)
        new_transformer_alphas_hybrid = dict()
        for predicate in transformer_alphas_hybrid:
            new_transformer_alphas_hybrid[predicate] = dict()
            for tau in transformer_alphas_hybrid[predicate]:
                new_transformer_alphas_hybrid[predicate][int(tau)] = dict()
                for agent in transformer_alphas_hybrid[predicate][tau]:
                    new_transformer_alphas_hybrid[predicate][int(tau)][int(agent)] = transformer_alphas_hybrid[predicate][tau][agent]
        transformer_alphas_hybrid = new_transformer_alphas_hybrid
    print("=== Finished Loading Alphas ===")
    print()

    # Perform experiment.
    print("=== Performing the Experiment ===")
    # Calculate delta_tilde.
    delta = 0.3
    delta_n = step_5_experiments.calculate_delta_n(delta, params.num_calibration_each_trial, step_5_experiments.f_divergence, epsilon)
    delta_tilde = step_5_experiments.calculate_delta_tilde(delta_n, step_5_experiments.f_divergence, epsilon)
    print("Delta Tilde:", delta_tilde)

    print("=== Conducting Direct Method ===")
    # Conduct direct method on the LSTM model.
    direct_nonconformity_list_lstm = step_4_distribution_shift_computation.calculate_direct_nonconformity(calib_trajectories, predicted_calib_trajectories_lstm)
    direct_nonconformity_list_lstm.sort()
    p_tilde_lstm = int(np.ceil(len(direct_nonconformity_list_lstm) * (1 - delta_tilde)))
    c_tilde_lstm = direct_nonconformity_list_lstm[p_tilde_lstm - 1]
    # Conduct direct method on the CNN model.
    direct_nonconformity_list_cnn = cnn.step_4_distribution_shift_computation.calculate_direct_nonconformity(calib_trajectories, predicted_calib_trajectories_cnn)
    direct_nonconformity_list_cnn.sort()
    p_tilde_cnn = int(np.ceil(len(direct_nonconformity_list_cnn) * (1 - delta_tilde)))
    c_tilde_cnn = direct_nonconformity_list_cnn[p_tilde_cnn - 1]
    # Conduct direct method on the Transformer model.
    direct_nonconformity_list_transformer = transformer.step_4_distribution_shift_computation.calculate_direct_nonconformity(calib_trajectories, predicted_calib_trajectories_transformer)
    direct_nonconformity_list_transformer.sort()
    p_tilde_transformer = int(np.ceil(len(direct_nonconformity_list_transformer) * (1 - delta_tilde)))
    c_tilde_transformer = direct_nonconformity_list_transformer[p_tilde_transformer - 1]
    min_value = min(np.concatenate((direct_nonconformity_list_lstm, direct_nonconformity_list_cnn, direct_nonconformity_list_transformer)))
    max_value = max(np.concatenate((direct_nonconformity_list_lstm, direct_nonconformity_list_cnn, direct_nonconformity_list_transformer)))
    y, x = np.histogram(direct_nonconformity_list_lstm,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="LSTM: Nonconformity Scores")
    y, x = np.histogram(direct_nonconformity_list_cnn,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="CNN: Nonconformity Scores")
    y, x = np.histogram(direct_nonconformity_list_transformer,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="Transformer: Nonconformity Scores")
    plt.axvline(c_tilde_lstm,  linestyle="--", color = "r", label="LSTM: $\\tilde{C}$")
    plt.axvline(c_tilde_cnn, linestyle="--", color = "g", label="CNN: $\\tilde{C}$")
    plt.axvline(c_tilde_transformer, linestyle="--", color = "b", label="Transformer: $\\tilde{C}$")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Nonconformity Score", fontsize=params.font_size)
    plt.ylabel("Frequency", fontsize=params.font_size)
    plt.legend(fontsize=params.legend_size)
    plt.tight_layout()
    plt.savefig("comparison_plots/direct_method_nonconformity.pdf")
    plt.show()
    # Perform testing
    direct_ground_robustnesses = []
    direct_worst_robustnesses_robust_lstm = []
    direct_worst_robustnesses_robust_cnn = []
    direct_worst_robustnesses_robust_transformer = []
    lstm_coverage_count = 0
    cnn_coverage_count = 0
    transformer_coverage_count = 0
    for index in random_indices_test_trajectories:
        history = test_trajectories[index]
        graph = step_1_data_analysis.DynamicGraph(history)
        direct_ground_robustness= graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
        pred_history_lstm = predicted_test_trajectories_lstm[index]
        pred_graph_lstm = step_1_data_analysis.DynamicGraph(pred_history_lstm)
        pred_robustness_lstm = pred_graph_lstm.calculate_final_robustness(0, params.T)[params.ego_agent][0]
        pred_history_cnn = predicted_test_trajectories_cnn[index]
        pred_graph_cnn = cnn.step_1_data_analysis.DynamicGraph(pred_history_cnn)
        pred_robustness_cnn = pred_graph_cnn.calculate_final_robustness(0, params.T)[params.ego_agent][0]
        pred_history_transformer = predicted_test_trajectories_transformer[index]
        pred_graph_transformer = transformer.step_1_data_analysis.DynamicGraph(pred_history_transformer)
        pred_robustness_transformer = pred_graph_transformer.calculate_final_robustness(0, params.T)[params.ego_agent][0]
        direct_worst_robustness_robust_lstm = pred_robustness_lstm - c_tilde_lstm
        direct_worst_robustness_robust_cnn = pred_robustness_cnn - c_tilde_cnn
        direct_worst_robustness_robust_transformer = pred_robustness_transformer - c_tilde_transformer
        direct_ground_robustnesses.append(direct_ground_robustness)
        direct_worst_robustnesses_robust_lstm.append(direct_worst_robustness_robust_lstm)
        direct_worst_robustnesses_robust_cnn.append(direct_worst_robustness_robust_cnn)
        direct_worst_robustnesses_robust_transformer.append(direct_worst_robustness_robust_transformer)
        if direct_ground_robustness >= direct_worst_robustness_robust_lstm:
            lstm_coverage_count += 1
        if direct_ground_robustness >= direct_worst_robustness_robust_cnn:
            cnn_coverage_count += 1
        if direct_ground_robustness >= direct_worst_robustness_robust_transformer:
            transformer_coverage_count += 1
    x_data = [i for i in range(len(direct_ground_robustnesses))]
    sorted_direct_ground_robustnesses, sorted_direct_worst_robustnesses_robust_lstm, sorted_direct_worst_robustnesses_robust_cnn, sorted_direct_worst_robustnesses_robust_transformer = zip(
        *sorted(zip(direct_ground_robustnesses, direct_worst_robustnesses_robust_lstm, direct_worst_robustnesses_robust_cnn, direct_worst_robustnesses_robust_transformer)))
    dot_sizes = [5 for i in range(len(x_data))]
    plt.scatter(x_data, sorted_direct_ground_robustnesses, s=dot_sizes, color='r',
                label='$\\rho^\psi(X, \\tau_0, l))$')
    plt.scatter(x_data, sorted_direct_worst_robustnesses_robust_lstm, s=dot_sizes, color='b',
                label='$\\rho^*$ (Robust Accurate Method with LSTM)')
    plt.scatter(x_data, sorted_direct_worst_robustnesses_robust_cnn, s=dot_sizes, color='g',
                label='$\\rho^*$ (Robust Accurate Method with CNN)')
    plt.scatter(x_data, sorted_direct_worst_robustnesses_robust_transformer, s=dot_sizes, color='c',
                label='$\\rho^*$ (Robust Accurate Method with Transformer)')
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Sample (Sorted on $\\rho^\psi(X, \\tau_0, l))$)", fontsize=params.font_size)
    plt.ylabel("Robust Semantics Value", fontsize=params.font_size)
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison_plots/direct_method_robustness.pdf")
    plt.show()
    print("Coverage for LSTM:", lstm_coverage_count / params.num_test_each_trial)
    print("Coverage for CNN:", cnn_coverage_count / params.num_test_each_trial)
    print("Coverage for Transformer:", transformer_coverage_count / params.num_test_each_trial)
    # Save coverages.
    with open("comparison_plots/direct_method_coverages.txt", "w") as f:
        f.write("Coverage for LSTM: " + str(lstm_coverage_count / params.num_test_each_trial) + "\n")
        f.write("Coverage for CNN: " + str(cnn_coverage_count / params.num_test_each_trial) + "\n")
        f.write("Coverage for Transformer: " + str(transformer_coverage_count / params.num_test_each_trial) + "\n")
    print("=== Finished Conducting Direct Method ===")

    print("=== Conducting Indirect Method ===")
    indirect_nonconformity_list_lstm = step_4_distribution_shift_computation.calculate_indirect_nonconformity(
        calib_trajectories, predicted_calib_trajectories_lstm, lstm_alphas_indirect)
    indirect_nonconformity_list_lstm.sort()
    p_tilde_lstm = int(np.ceil(len(indirect_nonconformity_list_lstm) * (1 - delta_tilde)))
    c_tilde_lstm = indirect_nonconformity_list_lstm[p_tilde_lstm - 1]
    indirect_nonconformity_list_cnn = cnn.step_4_distribution_shift_computation.calculate_indirect_nonconformity(
        calib_trajectories, predicted_calib_trajectories_cnn, cnn_alphas_indirect)
    indirect_nonconformity_list_cnn.sort()
    p_tilde_cnn = int(np.ceil(len(indirect_nonconformity_list_cnn) * (1 - delta_tilde)))
    c_tilde_cnn = indirect_nonconformity_list_cnn[p_tilde_cnn - 1]
    indirect_nonconformity_list_transformer = transformer.step_4_distribution_shift_computation.calculate_indirect_nonconformity(
        calib_trajectories, predicted_calib_trajectories_transformer, transformer_alphas_indirect)
    indirect_nonconformity_list_transformer.sort()
    p_tilde_transformer = int(np.ceil(len(indirect_nonconformity_list_transformer) * (1 - delta_tilde)))
    c_tilde_transformer = indirect_nonconformity_list_transformer[p_tilde_transformer - 1]
    min_value = min(np.concatenate((indirect_nonconformity_list_lstm, indirect_nonconformity_list_cnn, indirect_nonconformity_list_transformer)))
    max_value = max(np.concatenate((indirect_nonconformity_list_lstm, indirect_nonconformity_list_cnn, indirect_nonconformity_list_transformer)))
    y, x = np.histogram(indirect_nonconformity_list_lstm,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="LSTM: Nonconformity Scores")
    y, x = np.histogram(indirect_nonconformity_list_cnn,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="CNN: Nonconformity Scores")
    y, x = np.histogram(indirect_nonconformity_list_transformer,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="Transformer: Nonconformity Scores")
    plt.axvline(c_tilde_lstm, linestyle="--", color="r", label="LSTM: $\\tilde{C}$")
    plt.axvline(c_tilde_cnn, linestyle="--", color="g", label="CNN: $\\tilde{C}$")
    plt.axvline(c_tilde_transformer, linestyle="--", color="b", label="Transformer: $\\tilde{C}$")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Nonconformity Score", fontsize=params.font_size)
    plt.ylabel("Frequency", fontsize=params.font_size)
    plt.legend(fontsize=params.legend_size)
    plt.savefig("comparison_plots/indirect_method_nonconformity.pdf")
    plt.tight_layout()
    plt.show()
    # Generate the prediction regions.
    indirect_prediction_region_robust_lstm = dict()
    for tau in range(params.current_time + 1, params.T + 1):
        indirect_prediction_region_robust_lstm[tau] = dict()
        for agent in range(params.num_agents):
            indirect_prediction_region_robust_lstm[tau][agent] = c_tilde_lstm * lstm_alphas_indirect[tau][agent]
    indirect_prediction_region_robust_cnn = dict()
    for tau in range(params.current_time + 1, params.T + 1):
        indirect_prediction_region_robust_cnn[tau] = dict()
        for agent in range(params.num_agents):
            indirect_prediction_region_robust_cnn[tau][agent] = c_tilde_cnn * cnn_alphas_indirect[tau][agent]
    indirect_prediction_region_robust_transformer = dict()
    for tau in range(params.current_time + 1, params.T + 1):
        indirect_prediction_region_robust_transformer[tau] = dict()
        for agent in range(params.num_agents):
            indirect_prediction_region_robust_transformer[tau][agent] = c_tilde_transformer * transformer_alphas_indirect[tau][agent]
    # Plot prediction regions.
    print("length,", len(calib_indirect_nonconformity_2023_cnn))
    min_value = min(np.concatenate((calib_indirect_nonconformity_2023_lstm, calib_indirect_nonconformity_2023_cnn, calib_indirect_nonconformity_2023_transformer)))
    max_value = max(np.concatenate((calib_indirect_nonconformity_2023_lstm, calib_indirect_nonconformity_2023_cnn, calib_indirect_nonconformity_2023_transformer)))
    y, x = np.histogram(calib_indirect_nonconformity_2023_lstm,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y, color = "r")
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, color="r", label="LSTM: $\|X^{(i)}[l'] - \hat{X}^{(i)}[l']\|$")
    y, x = np.histogram(calib_indirect_nonconformity_2023_cnn,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y, color = "g")
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, color="g", label="CNN: $\|X^{(i)}[l'] - \hat{X}^{(i)}[l']\|$")
    y, x = np.histogram(calib_indirect_nonconformity_2023_transformer,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y, color = "b")
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, color="b", label="Transformer: $\|X^{(i)}[l'] - \hat{X}^{(i)}[l']\|$")
    plt.axvline(indirect_prediction_region_robust_lstm[experiment_tau][experiment_agent], linestyle="--", color="r", label="LSTM: $\\tilde{C}\\alpha_{\\tau, l'}$")
    plt.axvline(indirect_prediction_region_robust_cnn[experiment_tau][experiment_agent], linestyle="--", color="g", label="CNN: $\\tilde{C}\\alpha_{\\tau, l'}$")
    plt.axvline(indirect_prediction_region_robust_transformer[experiment_tau][experiment_agent], linestyle="--", color="b", label="Transformer: $\\tilde{C}\\alpha_{\\tau, l'}$")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Prediction Residual", fontsize=params.font_size)
    plt.ylabel("Frequency", fontsize=params.font_size)
    plt.legend(fontsize=params.legend_size)
    plt.tight_layout()
    plt.savefig("comparison_plots/indirect_method_prediction_regions.pdf")
    plt.show()
    # Perform testing
    indirect_ground_robustnesses = []
    indirect_worst_robustnesses_robust_lstm = []
    indirect_worst_robustnesses_robust_cnn = []
    indirect_worst_robustnesses_robust_transformer = []
    lstm_coverage_count = 0
    cnn_coverage_count = 0
    transformer_coverage_count = 0
    testing_progress = 0
    for index in random_indices_test_trajectories:
        if (testing_progress + 1) % 10 == 0:
            print("Testing on test index:", testing_progress + 1, "out of", params.num_test_each_trial, "data points.")
        history = test_trajectories[index]
        graph = step_1_data_analysis.DynamicGraph(history)
        indirect_ground_robustness = graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
        pred_history_lstm = predicted_test_trajectories_lstm[index]
        pred_graph_lstm = step_1_data_analysis.DynamicGraph(pred_history_lstm)
        indirect_worst_robustness_robust_lstm = step_5_experiments.compute_worst_case_robustness_indirect(pred_graph_lstm, indirect_prediction_region_robust_lstm)
        pred_history_cnn = predicted_test_trajectories_cnn[index]
        pred_graph_cnn = cnn.step_1_data_analysis.DynamicGraph(pred_history_cnn)
        indirect_worst_robustness_robust_cnn = step_5_experiments.compute_worst_case_robustness_indirect(pred_graph_cnn, indirect_prediction_region_robust_cnn)
        pred_history_transformer = predicted_test_trajectories_transformer[index]
        pred_graph_transformer = transformer.step_1_data_analysis.DynamicGraph(pred_history_transformer)
        indirect_worst_robustness_robust_transformer = step_5_experiments.compute_worst_case_robustness_indirect(pred_graph_transformer, indirect_prediction_region_robust_transformer)
        indirect_ground_robustnesses.append(indirect_ground_robustness)
        indirect_worst_robustnesses_robust_lstm.append(indirect_worst_robustness_robust_lstm)
        indirect_worst_robustnesses_robust_cnn.append(indirect_worst_robustness_robust_cnn)
        indirect_worst_robustnesses_robust_transformer.append(indirect_worst_robustness_robust_transformer)
        if indirect_ground_robustness >= indirect_worst_robustness_robust_lstm:
            lstm_coverage_count += 1
        if indirect_ground_robustness >= indirect_worst_robustness_robust_cnn:
            cnn_coverage_count += 1
        if indirect_ground_robustness >= indirect_worst_robustness_robust_transformer:
            transformer_coverage_count += 1
        testing_progress += 1
    x_data = [i for i in range(len(indirect_ground_robustnesses))]
    sorted_indirect_ground_robustnesses, sorted_indirect_worst_robustnesses_robust_lstm, sorted_indirect_worst_robustnesses_robust_cnn, sorted_indirect_worst_robustnesses_robust_transformer = zip(
        *sorted(zip(indirect_ground_robustnesses, indirect_worst_robustnesses_robust_lstm, indirect_worst_robustnesses_robust_cnn, indirect_worst_robustnesses_robust_transformer)))
    dot_sizes = [5 for i in range(len(x_data))]
    plt.scatter(x_data, sorted_indirect_ground_robustnesses, s=dot_sizes, color='r',
                label='$\\rho^\psi(X, \\tau_0, l))$')
    plt.scatter(x_data, sorted_indirect_worst_robustnesses_robust_lstm, s=dot_sizes, color='b',
                label='$\\rho^*$ (Robust Interpretable Method Variant I with LSTM)')
    plt.scatter(x_data, sorted_indirect_worst_robustnesses_robust_cnn, s=dot_sizes, color='g',
                label='$\\rho^*$ (Robust Interpretable Method Variant I with CNN)')
    plt.scatter(x_data, sorted_indirect_worst_robustnesses_robust_transformer, s=dot_sizes, color='c',
                label='$\\rho^*$ (Robust Interpretable Method Variant I with Transformer)')
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Sample (Sorted on $\\rho^\psi(X, \\tau_0, l))$", fontsize=params.font_size)
    plt.ylabel("Robust Semantics Value", fontsize=params.font_size)
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison_plots/indirect_method_robustness.pdf")
    plt.show()
    print("Coverage for LSTM:", lstm_coverage_count / params.num_test_each_trial)
    print("Coverage for CNN:", cnn_coverage_count / params.num_test_each_trial)
    print("Coverage for Transformer:", transformer_coverage_count / params.num_test_each_trial)
    # Save coverages.
    with open("comparison_plots/indirect_method_coverages.txt", "w") as f:
        f.write("Coverage for LSTM: " + str(lstm_coverage_count / params.num_test_each_trial) + "\n")
        f.write("Coverage for CNN: " + str(cnn_coverage_count / params.num_test_each_trial) + "\n")
        f.write("Coverage for Transformer: " + str(transformer_coverage_count / params.num_test_each_trial) + "\n")
    print("=== Finished Conducting Indirect Method ===")

    print("=== Conducting Hybrid Method ===")
    hybrid_nonconformity_list_lstm = step_4_distribution_shift_computation.calculate_hybrid_nonconformity(
        calib_trajectories, predicted_calib_trajectories_lstm, lstm_alphas_hybrid)
    hybrid_nonconformity_list_lstm.sort()
    p_tilde_lstm = int(np.ceil(len(hybrid_nonconformity_list_lstm) * (1 - delta_tilde)))
    c_tilde_lstm = hybrid_nonconformity_list_lstm[p_tilde_lstm - 1]
    hybrid_nonconformity_list_cnn = cnn.step_4_distribution_shift_computation.calculate_hybrid_nonconformity(
        calib_trajectories, predicted_calib_trajectories_cnn, cnn_alphas_hybrid)
    hybrid_nonconformity_list_cnn.sort()
    p_tilde_cnn = int(np.ceil(len(hybrid_nonconformity_list_cnn) * (1 - delta_tilde)))
    c_tilde_cnn = hybrid_nonconformity_list_cnn[p_tilde_cnn - 1]
    hybrid_nonconformity_list_transformer = transformer.step_4_distribution_shift_computation.calculate_hybrid_nonconformity(
        calib_trajectories, predicted_calib_trajectories_transformer, transformer_alphas_hybrid)
    hybrid_nonconformity_list_transformer.sort()
    p_tilde_transformer = int(np.ceil(len(hybrid_nonconformity_list_transformer) * (1 - delta_tilde)))
    c_tilde_transformer = hybrid_nonconformity_list_transformer[p_tilde_transformer - 1]
    min_value = min(np.concatenate((hybrid_nonconformity_list_lstm, hybrid_nonconformity_list_cnn, hybrid_nonconformity_list_transformer)))
    max_value = max(np.concatenate((hybrid_nonconformity_list_lstm, hybrid_nonconformity_list_cnn, hybrid_nonconformity_list_transformer)))
    y, x = np.histogram(hybrid_nonconformity_list_lstm,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="LSTM: Nonconformity Scores")
    y, x = np.histogram(hybrid_nonconformity_list_cnn,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="CNN: Nonconformity Scores")
    y, x = np.histogram(hybrid_nonconformity_list_transformer,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="Transformer: Nonconformity Scores")
    plt.axvline(c_tilde_lstm, linestyle="--", color="r", label="LSTM: $\\tilde{C}$")
    plt.axvline(c_tilde_cnn, linestyle="--", color="g", label="CNN: $\\tilde{C}$")
    plt.axvline(c_tilde_transformer, linestyle="--", color="b", label="Transformer: $\\tilde{C}$")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Nonconformity Score", fontsize=params.font_size)
    plt.ylabel("Frequency", fontsize=params.font_size)
    plt.legend(fontsize=params.legend_size)
    plt.tight_layout()
    plt.savefig("comparison_plots/hybrid_method_nonconformity.pdf")
    plt.show()
    # Plot prediction regions.
    print("length,", len(calib_hybrid_nonconformity_2023_cnn))
    min_value = min(np.concatenate((calib_hybrid_nonconformity_2023_lstm, calib_hybrid_nonconformity_2023_cnn, calib_hybrid_nonconformity_2023_transformer)))
    max_value = max(np.concatenate((calib_hybrid_nonconformity_2023_lstm, calib_hybrid_nonconformity_2023_cnn, calib_hybrid_nonconformity_2023_transformer)))
    y, x = np.histogram(calib_hybrid_nonconformity_2023_lstm,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y, color="r")
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, color="r", label="LSTM: $\\rho^\pi(\hat{X}^{(i)}) - \\rho^\pi(X^{(i)})$")
    y, x = np.histogram(calib_hybrid_nonconformity_2023_cnn,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y, color="g")
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, color="g", label="CNN: $\\rho^\pi(\hat{X}^{(i)}) - \\rho^\pi(X^{(i)})$")
    y, x = np.histogram(calib_hybrid_nonconformity_2023_transformer,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y, color="b")
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, color="b", label="Transformer: $\\rho^\pi(\hat{X}^{(i)}) - \\rho^\pi(X^{(i)})$")
    plt.axvline(c_tilde_lstm * lstm_alphas_hybrid[experiment_pi][experiment_tau][experiment_agent], linestyle="--", color="r", label="LSTM: $\\tilde{C}\\alpha_{\pi, \\tau, l'}$")
    plt.axvline(c_tilde_cnn * cnn_alphas_hybrid[experiment_pi][experiment_tau][experiment_agent], linestyle="--", color="g", label="CNN: $\\tilde{C}\\alpha_{\pi, \\tau, l'}$")
    plt.axvline(c_tilde_transformer * transformer_alphas_hybrid[experiment_pi][experiment_tau][experiment_agent], linestyle="--", color="b", label="Transformer: $\\tilde{C}\\alpha_{\pi, \\tau, l'}$")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("$\\tilde{C}\\alpha_{\pi, \\tau, l'}$", fontsize=params.font_size)
    plt.ylabel("Frequency", fontsize=params.font_size)
    plt.legend(fontsize =  params.legend_size)
    plt.tight_layout()
    plt.savefig("comparison_plots/hybrid_method_prediction_regions.pdf")
    plt.show()
    # Calculate the coverage for the hybrid methods.
    testing_progress = 0
    hybrid_ground_robustnesses = []
    hybrid_worst_robustnesses_robust_lstm = []
    hybrid_worst_robustnesses_robust_cnn = []
    hybrid_worst_robustnesses_robust_transformer = []
    lstm_coverage_count = 0
    cnn_coverage_count = 0
    transformer_coverage_count = 0
    for index in random_indices_test_trajectories:
        if (testing_progress + 1) % 10 == 0:
            print("Testing on test index:", testing_progress + 1, "out of", params.num_test_each_trial,
                  "data points.")
        history = test_trajectories[index]
        graph = step_1_data_analysis.DynamicGraph(history)
        hybrid_ground_robustness = graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
        pred_history_lstm = predicted_test_trajectories_lstm[index]
        pred_graph_lstm = step_1_data_analysis.DynamicGraph(pred_history_lstm)
        hybrid_worst_robustness_robust_lstm = step_5_experiments.compute_worst_case_robustness_hybrid(pred_graph_lstm, lstm_alphas_hybrid, c_tilde_lstm)
        pred_history_cnn = predicted_test_trajectories_cnn[index]
        pred_graph_cnn = cnn.step_1_data_analysis.DynamicGraph(pred_history_cnn)
        hybrid_worst_robustness_robust_cnn = step_5_experiments.compute_worst_case_robustness_hybrid(pred_graph_cnn, cnn_alphas_hybrid, c_tilde_cnn)
        pred_history_transformer = predicted_test_trajectories_transformer[index]
        pred_graph_transformer = transformer.step_1_data_analysis.DynamicGraph(pred_history_transformer)
        hybrid_worst_robustness_robust_transformer = step_5_experiments.compute_worst_case_robustness_hybrid(pred_graph_transformer, transformer_alphas_hybrid, c_tilde_transformer)
        hybrid_ground_robustnesses.append(hybrid_ground_robustness)
        hybrid_worst_robustnesses_robust_lstm.append(hybrid_worst_robustness_robust_lstm)
        hybrid_worst_robustnesses_robust_cnn.append(hybrid_worst_robustness_robust_cnn)
        hybrid_worst_robustnesses_robust_transformer.append(hybrid_worst_robustness_robust_transformer)
        if hybrid_ground_robustness >= hybrid_worst_robustness_robust_lstm:
            lstm_coverage_count += 1
        if hybrid_ground_robustness >= hybrid_worst_robustness_robust_cnn:
            cnn_coverage_count += 1
        if hybrid_ground_robustness >= hybrid_worst_robustness_robust_transformer:
            transformer_coverage_count += 1
        testing_progress += 1
    x_data = [i for i in range(len(hybrid_ground_robustnesses))]
    sorted_hybrid_ground_robustnesses, sorted_hybrid_worst_robustnesses_robust_lstm, sorted_hybrid_worst_robustnesses_robust_cnn, sorted_hybrid_worst_robustnesses_robust_transformer = zip(
        *sorted(zip(hybrid_ground_robustnesses, hybrid_worst_robustnesses_robust_lstm, hybrid_worst_robustnesses_robust_cnn, hybrid_worst_robustnesses_robust_transformer)))
    dot_sizes = [5 for i in range(len(x_data))]
    plt.scatter(x_data, sorted_hybrid_ground_robustnesses, s=dot_sizes, color='r',
                label='$\\rho^\psi(X, \\tau_0, l))$')
    plt.scatter(x_data, sorted_hybrid_worst_robustnesses_robust_lstm, s=dot_sizes, color='b',
                label='$\\rho^*$ (Robust Interpretable Method Variant II with LSTM)')
    plt.scatter(x_data, sorted_hybrid_worst_robustnesses_robust_cnn, s=dot_sizes, color='g',
                label='$\\rho^*$ (Robust Interpretable Method Variant II with CNN)')
    plt.scatter(x_data, sorted_hybrid_worst_robustnesses_robust_transformer, s=dot_sizes, color='c',
                label='$\\rho^*$ (Robust Interpretable Method Variant II with Transformer)')
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Sample (Sorted on $\\rho^\psi(X, \\tau_0, l))$", fontsize=params.font_size)
    plt.ylabel("Robust Semantics Value", fontsize=params.font_size)
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison_plots/hybrid_method_robustness.pdf")
    plt.show()
    print("Coverage for LSTM:", lstm_coverage_count / params.num_test_each_trial)
    print("Coverage for CNN:", cnn_coverage_count / params.num_test_each_trial)
    print("Coverage for Transformer:", transformer_coverage_count / params.num_test_each_trial)
    # Save coverages.
    with open("comparison_plots/hybrid_method_coverages.txt", "w") as f:
        f.write("Coverage for LSTM: " + str(lstm_coverage_count / params.num_test_each_trial) + "\n")
        f.write("Coverage for CNN: " + str(cnn_coverage_count / params.num_test_each_trial) + "\n")
        f.write("Coverage for Transformer: " + str(transformer_coverage_count / params.num_test_each_trial) + "\n")
    print("=== Finished Conducting Hybrid Method ===")

    print("=== Finished Performing the Experiment ===")

if __name__ == "__main__":
    main()