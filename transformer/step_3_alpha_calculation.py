import step_0_data_processing
import step_2_predictor_training
import step_1_data_analysis
import params
import keras
import json
import time


def closest_euclidean_distance_to_obstacles(x):
    centers, side_length = step_1_data_analysis. process_obstacles()
    closest_distance = float("inf")
    for center in centers:
        new_distance = max(abs(x[0] - center[0]),
                           abs(x[1] - center[1])) - side_length / 2
        closest_distance = min(closest_distance, new_distance)
    return closest_distance


def main():
    # First, load the trajectories.
    trajectories = step_0_data_processing.load_trajectories()
    # Load training trajectories.
    training_trajectories = dict()
    for i in range(1, params.num_train + 1):
        training_trajectories[i] = trajectories[i]
    # Load the Transformer models.
    print("=== Loading the Transformer Model ===")
    trained_transformer_models = dict()
    for a in range(params.num_agents):
        trained_transformer_models[a] = dict()
        for s in range(3):
            trained_transformer_models[a][s] = keras.models.load_model(
                f"predictors/{params.num_agents}-agent/transformer/transformer_model_agent_{a}_state_{s}.keras")
    print("=== Finished Loading the Transformer Model ===")
    with open(f'predictors/{params.num_agents}-agent/transformer/norm.txt', 'r') as f:
        transformer_norm = float(f.read())
    # Find the predicted trajectories based on the training trajectories.
    predicted_trajectories = step_2_predictor_training.generate_pred_trajectories(trained_transformer_models, training_trajectories, transformer_norm)
    print("=== Calculating alphas for the indirect method ===")
    # Time for the calculations.
    alphas_indirect_start_time = time.time()
    alphas_indirect = dict()
    # Calculate alphas for the indirect method.
    for tau in range(params.current_time + 1, params.T + 1):
        alphas_indirect[tau] = dict()
        for agent in range(params.num_agents):
            alpha_list = []
            for i in range(1, params.num_train + 1):
                x = training_trajectories[i][agent][tau]
                x_hat = predicted_trajectories[i][agent][tau]
                # Calculate the L_2 norm between x and x_hat.
                alpha = ((x[0] - x_hat[0]) ** 2 + (x[1] - x_hat[1]) ** 2 + (x[2] - x_hat[2]) ** 2) ** (1 / 2)
                alpha_list.append(alpha)
            alphas_indirect[tau][agent] = max(alpha_list)
    alphas_indirect_time = time.time() - alphas_indirect_start_time
    # Save the alphas.
    with open(f"alphas/{params.num_agents}-agent/alphas_indirect.json", "w") as f:
        json.dump(alphas_indirect, f)
    print("=== Finished calculating alphas for the indirect method ===")
    print("=== Calculating alphas for the hybrid method ===")
    # Time for calculations.
    alphas_hybrid_start_time = time.time()
    # Calculate alphas for the hybrid method.
    alphas_hybrid = dict()
    # Calculate alphas
    alphas_hybrid["communication_to_terminal_check"] = dict()
    alphas_hybrid["ground_collision_avoidance_check"] = dict()
    alphas_hybrid["goal_reaching"] = dict()
    alphas_hybrid["closest_euclidean_distance_to_obstacles"] = dict()
    for tau in range(params.current_time + 1, params.T + 1):
        alphas_hybrid["communication_to_terminal_check"][tau] = dict()
        alphas_hybrid["ground_collision_avoidance_check"][tau] = dict()
        alphas_hybrid["goal_reaching"][tau] = dict()
        alphas_hybrid["closest_euclidean_distance_to_obstacles"][tau] = dict()
        for agent in range(params.num_agents):
            alpha_communication_list = []
            alpha_ground_collision_list = []
            alphas_goal_reaching_list = []
            alphas_obstacles_list = []
            for i in range(1, params.num_train + 1):
                x_hat = predicted_trajectories[i][agent][tau]
                x = training_trajectories[i][agent][tau]

                rho_x_hat_communication = params.terminal_height - x_hat[2]
                rho_x_communication = params.terminal_height - x[2]
                rho_x_hat_ground_collision = x_hat[2] - params.ground_height
                rho_x_ground_collision = x[2] - params.ground_height
                rho_x_hat_goal_reaching = x_hat[0] - params.goal_location
                rho_x_goal_reaching = x[0] - params.goal_location
                rho_x_hat_obstacle = closest_euclidean_distance_to_obstacles(x_hat)
                rho_x_obstacle = closest_euclidean_distance_to_obstacles(x)

                alpha_communication_list.append(abs(rho_x_hat_communication - rho_x_communication))
                alpha_ground_collision_list.append(abs(rho_x_hat_ground_collision - rho_x_ground_collision))
                alphas_goal_reaching_list.append(abs(rho_x_hat_goal_reaching - rho_x_goal_reaching))
                alphas_obstacles_list.append(abs(rho_x_hat_obstacle - rho_x_obstacle))
            alphas_hybrid["communication_to_terminal_check"][tau][agent] = max(alpha_communication_list)
            alphas_hybrid["ground_collision_avoidance_check"][tau][agent] = max(alpha_ground_collision_list)
            alphas_hybrid["goal_reaching"][tau][agent] = max(alphas_goal_reaching_list)
            alphas_hybrid["closest_euclidean_distance_to_obstacles"][tau][agent] = max(alphas_obstacles_list)
    alphas_hybrid_time = time.time() - alphas_hybrid_start_time
    # Save the alphas.
    with open(f"alphas/{params.num_agents}-agent/alphas_hybrid.json", "w") as f:
        json.dump(alphas_hybrid, f)
    # Save the times.
    with open(f"alphas/{params.num_agents}-agent/alphas_indirect_time.txt", "w") as f:
        f.write(str(alphas_indirect_time))
    with open(f"alphas/{params.num_agents}-agent/alphas_hybrid_time.txt", "w") as f:
        f.write(str(alphas_hybrid_time))
    print("=== Finished calculating alphas for the hybrid method ===")


if __name__ == "__main__":
    main()