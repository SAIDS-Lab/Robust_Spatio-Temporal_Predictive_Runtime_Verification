import params
import numpy as np
import step_0_data_processing
import step_1_data_analysis
import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
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


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def train_transformer_models(x_train_whole, y_train_whole):
    print("=== Training transformer Models ===")
    trained_transformer_models = dict()
    for a in range(params.num_agents):
        trained_transformer_models[a] = dict()
        for s in range(3):
            print("Training transformer Model for Agent " + str(a) + " and State " + str(s) + ".")
            if s == 0:
                epoch_size = 200
            elif s == 1:
                epoch_size = 250
            else:
                epoch_size = 200
            # Create a Keras Model.
            # Reshape x_train to (num_samples, params.current_time + 1, 1)
            x_train = x_train_whole[a][s]
            y_train = y_train_whole[a][s]
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

            input_shape = x_train.shape[1:]

            model = build_model(
                input_shape,
                head_size=256,
                num_heads=4,
                ff_dim=4,
                num_transformer_blocks=10, # Use 10 transformer blocks.
                mlp_units=[256],
                output_shape=params.T - params.current_time,
                mlp_dropout=0,
                dropout=0,
            )
            model.compile(
                loss="mean_squared_error",  # Loss function for regression
                optimizer=optimizers.Adam(learning_rate=1e-4),
                metrics=["mean_squared_error"],  # Metric for regression
            )
            model.summary()

            callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

            model.fit(
                x_train,
                y_train,
                epochs=epoch_size,
                batch_size=64,
                callbacks=callbacks,
            )

            trained_transformer_models[a][s] = model
            model.save(f"predictors/{params.num_agents}-agent/transformer/transformer_model_agent_{a}_state_{s}.keras")
    print("=== Finished Training and Saving transformer Models. ===")
    return trained_transformer_models


def build_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        output_shape,
        dropout=0,
        mlp_dropout=0,
):
    inputs = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inputs)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Use GlobalMaxPooling1D or GlobalAveragePooling1D for handling variable-length sequences
    x = layers.GlobalAveragePooling1D()(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    x = layers.Dense(500, activation = "relu")(x)  # Single unit for regression
    x = layers.Dense(50, activation = "relu")(x)
    outputs = layers.Dense(output_shape)(x) # Add an additional dense layer.
    return keras.Model(inputs, outputs)

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
    plt.legend(fontsize = params.legend_size)
    plt.tight_layout()
    if my_title != "":
        plt.title(my_title + " " + f"with Ground Robustness of {np.round(robustness_ground, 2)}" + f" and Predicted Robustness of {np.round(robustness_pred, 2)}", fontsize=8)
    if file_name != "":
        plt.savefig(f"plots/{params.num_agents}-agent/" + file_name + params.plotting_saving_format)
    plt.show()


def main():
    # First, load the trajectories.
    trajectories = step_0_data_processing.load_trajectories()
    # Load training trajectories.
    training_trajectories = dict()
    for i in range(1, params.num_train + 1):
        training_trajectories[i] = trajectories[i]
    # Preprocess the training trajectories.
    norm = find_norm(training_trajectories)  # Find norm.
    # Save the norm.
    with open(f"predictors/{params.num_agents}-agent/transformer/norm.txt", "w") as f:
        f.write(str(norm))
    x_train_whole, y_train_whole = load_trajectories_to_tensors(training_trajectories, norm)
    example_index = 210

    # trained_transformer_models = train_transformer_models(x_train_whole, y_train_whole)

    print("=== Loading the Transformer Model ===")
    trained_transformer_models = dict()
    for a in range(params.num_agents):
        trained_transformer_models[a] = dict()
        for s in range(3):
            trained_transformer_models[a][s] = keras.models.load_model(f"predictors/{params.num_agents}-agent/transformer/transformer_model_agent_{a}_state_{s}.keras")
    print("=== Finished Loading the Transformer Model ===")

    print("== Loading the norm ===")
    with open(f"predictors/{params.num_agents}-agent/transformer/norm.txt", "r") as f:
        norm = float(f.read())

    """
    Illustrate the predictions.
    """
    # Let's use the trained models to predict the future states on a selected calibration data.
    # Load calibration trajectories.
    print("=== Illustrating the Predictions ===")

    print("=== Illustrating for Transformer ===")
    calib_trajectories = dict()
    for i in range(params.num_train + 1, params.num_histories + 1):
        calib_trajectories[i] = trajectories[i]
    predicted_trajectories = generate_pred_trajectories(trained_transformer_models, calib_trajectories, norm)
    plot_predictions_2d(calib_trajectories[example_index], predicted_trajectories[example_index], "", "transformer_predictions_example_nominal")
    shifted_trajectories = step_0_data_processing.load_trajectories(shifted=True)
    # Generate predictions.
    predicted_shifted_trajectories = generate_pred_trajectories(trained_transformer_models, shifted_trajectories, norm)
    plot_predictions_2d(shifted_trajectories[example_index], predicted_shifted_trajectories[example_index], "", "transformer_predictions_example_shifted")
    print("=== End of Illustration for Transformer ===")

    print("=== Finished Illustrating the Predictions ===")



if __name__ == '__main__':
    main()