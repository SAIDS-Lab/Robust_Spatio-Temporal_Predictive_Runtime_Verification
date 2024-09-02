import params
import numpy as np
import step_0_data_processing
import step_1_data_analysis
import keras
import matplotlib.pyplot as plt


def find_norm(selected_trajectories):
    max_val = 0
    for i in selected_trajectories.keys():
        for a in range(params.num_agents):
            for t in range(len(selected_trajectories[i][a])):
                for s in range(3):
                    max_val = max(max_val, abs(selected_trajectories[i][a][t][s]))
    return max_val


def load_trajectories_to_tensors(selected_trajectories, norm):
    x, y = dict(), dict() # The first index will be the agent, and the second index will be the state dimension.
    for a in range(params.num_agents):
        x[a] = dict()
        y[a] = dict()
        for s in range(3):
            x_a_s = []
            y_a_s = []
            for i in sorted(selected_trajectories.keys()):
                x_a_s.append([selected_trajectories[i][a][t][s] / norm for t in range(params.current_time + 1)])
                y_a_s.append([selected_trajectories[i][a][t][s] / norm for t in range(params.current_time + 1, len(selected_trajectories[i][a]))])
            x[a][s] = np.array(x_a_s)
            y[a][s] = np.array(y_a_s)
    return x, y


def generate_pred_trajectories(trained_models, ground_truth_trajectories, norm):
    x = dict()
    for a in range(params.num_agents):
        x[a] = dict()
        for s in range(3):
            x_a_s = []
            for i in sorted(ground_truth_trajectories.keys()):
                x_a_s.append([ground_truth_trajectories[i][a][t][s] / norm for t in range(params.current_time + 1)])
            x[a][s] = np.array(x_a_s)
    pred = dict()
    for a in range(params.num_agents):
        pred[a] = dict()
        for s in range(3):
            pred[a][s] = trained_models[a][s].predict(x[a][s])
    # Assemble pred back to the trajectory format.
    pred_trajectories = dict()
    i = 0
    for j in sorted(ground_truth_trajectories.keys()):
        pred_trajectories[j] = dict()
        for a in range(params.num_agents):
            pred_trajectories[j][a] = []
            for t in range(params.current_time + 1):
                pred_trajectories[j][a].append(ground_truth_trajectories[j][a][t])
            for t in range(len(pred[a][0][0])):
                pred_trajectories[j][a].append([float(pred[a][0][i][t]) * norm, float(pred[a][1][i][t]) * norm, float(pred[a][2][i][t]) * norm])
        i += 1
    return pred_trajectories


def plot_predictions_2d(ground_history, pred_history, my_title, file_name):
    # Plot the ground truth with solide line and the pred with dashed line.
    for a in range(params.num_agents):
        x = [s[0] for s in ground_history[a]]
        y = [s[1] for s in ground_history[a]]
        plt.plot(x, y, label = params.legend[a], color = params.colors[a])
        x = [s[0] for s in pred_history[a]]
        y = [s[1] for s in pred_history[a]]
        plt.plot(x, y, linestyle="--", color = params.colors[a])
    # Draw the obstacles.
    for obstacle in params.obstacles:
        plt.plot([obstacle[0][0], obstacle[1][0]], [obstacle[0][1], obstacle[1][1]], 'bo', linestyle="solid")
        plt.plot([obstacle[1][0], obstacle[3][0]], [obstacle[1][1], obstacle[3][1]], 'bo', linestyle="solid")
        plt.plot([obstacle[2][0], obstacle[3][0]], [obstacle[2][1], obstacle[3][1]], 'bo', linestyle="solid")
        plt.plot([obstacle[0][0], obstacle[2][0]], [obstacle[0][1], obstacle[2][1]], 'bo', linestyle="solid")

    # Calculate the final roubstness values for ground truth and predictions.
    ground_graph = step_1_data_analysis.DynamicGraph(ground_history)
    robustness_ground = ground_graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
    pred_graph = step_1_data_analysis.DynamicGraph(pred_history)
    robustness_pred = pred_graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]

    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("X", fontsize=params.label_size)
    plt.ylabel("Y", fontsize=params.label_size)
    plt.legend(fontsize=params.legend_size)
    plt.tight_layout()
    if my_title != "":
        plt.title(my_title + " " + f"with Ground Robustness of {np.round(robustness_ground, 2)}" + f" and Predicted Robustness of {np.round(robustness_pred, 2)}", fontsize=8)
    if file_name != "":
        plt.savefig(f"plots/{params.num_agents}-agent/" + file_name + params.plotting_saving_format)
    plt.show()


def main():
    # First, load the trajectories.
    trajectories = step_0_data_processing.load_trajectories()
    # Example index for illustration.
    example_index = 210

    print("=== Loading the CNN Model ===")
    trained_cnn_models = dict()
    for a in range(params.num_agents):
        trained_cnn_models[a] = dict()
        for s in range(3):
            trained_cnn_models[a][s] = keras.models.load_model(f"predictors/{params.num_agents}-agent/cnn/cnn_model_agent_{a}_state_{s}.keras")
    print("=== Finished Loading the CNN Model ===")

    print("== Loading the norm ===")
    with open(f"predictors/{params.num_agents}-agent/cnn/norm.txt", "r") as f:
        norm = float(f.read())

    """
    Illustrate the predictions.
    """
    # Let's use the trained models to predict the future states on a selected calibration data.
    # Load calibration trajectories.
    print("=== Illustrating the Predictions ===")

    print("=== Illustrating for CNN ===")
    calib_trajectories = dict()
    for i in range(params.num_train + 1, params.num_histories + 1):
        calib_trajectories[i] = trajectories[i]
    predicted_trajectories = generate_pred_trajectories(trained_cnn_models, calib_trajectories, norm)
    plot_predictions_2d(calib_trajectories[example_index], predicted_trajectories[example_index], "", "cnn_predictions_example_nominal")
    shifted_trajectories = step_0_data_processing.load_trajectories(shifted=True)
    # Generate predictions.
    predicted_shifted_trajectories = generate_pred_trajectories(trained_cnn_models, shifted_trajectories, norm)
    plot_predictions_2d(shifted_trajectories[example_index], predicted_shifted_trajectories[example_index], "", "cnn_predictions_example_shifted")
    print("=== End of Illustration for CNN ===")

    print("=== Finished Illustrating the Predictions ===")


if __name__ == '__main__':
    main()