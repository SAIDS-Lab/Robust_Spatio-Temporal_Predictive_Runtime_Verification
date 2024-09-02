import params
import numpy as np
import step_0_data_processing
import step_1_data_analysis
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Flatten
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


def generate_pred_trajectories(trained_cnn_models, ground_truth_trajectories, norm):
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
            pred[a][s] = trained_cnn_models[a][s].predict(x[a][s])
    # Assemble pred back to the trajectory format.
    pred_trajectories = dict()
    i = 0
    for j in sorted(ground_truth_trajectories.keys()):
        pred_trajectories[j] = dict()
        for a in range(params.num_agents):
            pred_trajectories[j][a] = []
            for t in range(params.current_time + 1):
                pred_trajectories[j][a].append(ground_truth_trajectories[j][a][t])
            for t in range(len(pred[a][0][0])): # a = num agents, 0 = dim, i = num of trajectories, t = time
                pred_trajectories[j][a].append([float(pred[a][0][i][t]) * norm, float(pred[a][1][i][t]) * norm, float(pred[a][2][i][t]) * norm])
        i += 1
    return pred_trajectories


def train_cnn_models(x_train_whole, y_train_whole):
    print("=== Training cnn Models ===")
    trained_cnn_models = dict()
    for a in range(params.num_agents):
        trained_cnn_models[a] = dict()
        for s in range(3):
            print("Training cnn Model for Agent " + str(a) + " and State " + str(s) + ".")
            # Create a Keras Model.
            cnn_model = Sequential()
            cnn_model.add(Conv1D(50, 1, activation="relu", input_shape=(params.current_time + 1, 1)))
            cnn_model.add(Flatten())
            cnn_model.add(Dense(100, activation="relu"))
            cnn_model.add(Dense(params.T - params.current_time))
            cnn_model.compile(loss='mse', optimizer='adam')
            # Fit network.
            x_train = x_train_whole[a][s]
            y_train = y_train_whole[a][s]
            
            # Reshape x_train to (num_samples, params.current_time + 1, 1)
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    
            cnn_model.fit(x_train, y_train, epochs=400, batch_size=1, verbose=0)
            #cnn_model.fit(x_train_whole[a][s], y_train_whole[a][s], epochs=50, batch_size=1, verbose=0)
            trained_cnn_models[a][s] = cnn_model
            # Save the model.
            cnn_model.save(f"predictors/cnn_400_epochs/cnn_model_agent_{a}_state_{s}.keras")
    print("=== Finished Training and Saving cnn Models.")
    return trained_cnn_models


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
        plt.plot([obstacle[0][0], obstacle[1][0]], [obstacle[0][1], obstacle[1][1]], 'bo', linestyle="--")
        plt.plot([obstacle[1][0], obstacle[3][0]], [obstacle[1][1], obstacle[3][1]], 'bo', linestyle="--")
        plt.plot([obstacle[2][0], obstacle[3][0]], [obstacle[2][1], obstacle[3][1]], 'bo', linestyle="--")
        plt.plot([obstacle[0][0], obstacle[2][0]], [obstacle[0][1], obstacle[2][1]], 'bo', linestyle="--")

    # Calculate the final roubstness values for ground truth and predictions.
    ground_graph = step_1_data_analysis.DynamicGraph(ground_history)
    robustness_ground = ground_graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
    pred_graph = step_1_data_analysis.DynamicGraph(pred_history)
    robustness_pred = pred_graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title(my_title + " " + f"with Ground Robustness of {np.round(robustness_ground, 2)}" + f" and Predicted Robustness of {np.round(robustness_pred, 2)}", fontsize = 8)
    if file_name != "":
        plt.savefig("plots/" + file_name + params.plotting_saving_format)
    plt.show()


def main():
    # First, load the trajectories.
    trajectories = step_0_data_processing.load_trajectories()
    # Load training trajectories.
    training_trajectories = dict()
    for i in range(1, params.num_train + 1):
        training_trajectories[i] = trajectories[i]
    # Preprocess the training trajectories.
    norm = find_norm(training_trajectories) # Find norm.
    # Save the norm.
    with open("predictors/cnn_400_epochs/norm.txt", "w") as f:
        f.write(str(norm))
    x_train_whole, y_train_whole = load_trajectories_to_tensors(training_trajectories, norm)
    # Example index for illustration.
    example_index = 210

    """
    Train an cnn Model for each of the dimension.
    """
    trained_cnn_models = train_cnn_models(x_train_whole, y_train_whole) # Uncomment if training the cnn models.
    # Load the trained cnn models.
    print("=== Loading the cnn Model ===")
    trained_cnn_models = dict()
    for a in range(params.num_agents):
        trained_cnn_models[a] = dict()
        for s in range(3):
            trained_cnn_models[a][s] = keras.models.load_model(f"predictors/cnn_400_epochs/cnn_model_agent_{a}_state_{s}.keras")
    print("=== Finished Loading the cnn Model ===")

    """
    Illustrate the predictions.
    """
    # Let's use the trained models to predict the future states on a selected calibration data.
    # Load calibration trajectories.
    print("=== Illustrating the Predictions ===")

    print("=== Illustrating for cnn ===")
    calib_trajectories = dict()
    for i in range(params.num_train + 1, params.num_histories + 1):
        calib_trajectories[i] = trajectories[i]
    print(len(calib_trajectories))
    predicted_trajectories = generate_pred_trajectories(trained_cnn_models, calib_trajectories, norm) # PROBLEM
    plot_predictions_2d(calib_trajectories[example_index], predicted_trajectories[example_index], "Example Prediction from cnn", "cnn__predictions_example")
    # Show histograms of ground truth robustness vs. predicted robustness.
    ground_robustnesses = []
    pred_robustnesses = []
    for i in range(params.num_train + 1, params.num_histories + 1):
        if i % 50 == 0:
            print("Analyzing robustness for index:", i)
        ground_graph = step_1_data_analysis.DynamicGraph(calib_trajectories[i])
        robustness_ground = ground_graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
        pred_graph = step_1_data_analysis.DynamicGraph(predicted_trajectories[i])
        robustness_pred = pred_graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
        ground_robustnesses.append(robustness_ground)
        pred_robustnesses.append(robustness_pred)
    plt.hist(ground_robustnesses, bins=100, alpha=0.5, label='Ground Truth')
    plt.hist(pred_robustnesses, bins=100, alpha=0.5, label='Predicted')
    plt.legend(loc='upper right')
    plt.title("Histogram of Robustnesses for cnn")
    plt.xlabel("Robustness")
    plt.ylabel("Frequency")
    plt.savefig("plots/cnn_histogram_robustnesses.png")
    plt.show()
    print("=== End of Illustration for cnn ===")

    print("=== Finished Illustrating the Predictions ===")


if __name__ == '__main__':
    main()