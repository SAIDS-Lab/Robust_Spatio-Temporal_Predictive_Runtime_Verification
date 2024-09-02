import step_2_predictor_training
import step_0_data_processing
import step_1_data_analysis
import params
import keras
import json


def evaluate_ground_collision_avoidance(history, t):
    return history[t][2] - params.ground_height


def evaluate_obstacle_avoidance(history, t):
    centers, side_length = step_1_data_analysis.process_obstacles()
    closest_distance = float("inf")
    for center in centers:
        new_distance = max(abs(history[t][0] - center[0]),
                           abs(history[t][1] - center[1])) - side_length / 2
        closest_distance = min(closest_distance, new_distance)
    return closest_distance


def evaluate_goal_reaching(history, t):
    return history[t][0] - params.goal_location

def evaluate_final_robustness(history):
    robustness_1 = float("inf")
    for t in range(0, params.T + 1):
        robustness_1 = min(robustness_1, min(evaluate_ground_collision_avoidance(history, t), evaluate_obstacle_avoidance(history, t)))
    robustness_2 = -float("inf")
    for t in range(0, params.T + 1):
        robustness_2 = max(robustness_2, evaluate_goal_reaching(history, t))
    return min(robustness_1, robustness_2)


def main():
    # First, load the trajectories.
    trajectories = step_0_data_processing.load_trajectories()
    # Load training trajectories.
    training_trajectories = dict()
    for i in range(1, params.num_train + 1):
        training_trajectories[i] = trajectories[i]
    # Load norm.
    with open(f'predictors/{params.num_agents}-agent/lstm/norm.txt', 'r') as f:
        lstm_norm = float(f.read())
    # Load predictor.
    print("=== Loading the LSTM Model ===")
    trained_lstm_models = dict()
    for a in range(params.num_agents):
        trained_lstm_models[a] = dict()
        for s in range(3):
            trained_lstm_models[a][s] = keras.models.load_model(
                f"predictors/{params.num_agents}-agent/lstm/lstm_model_agent_{a}_state_{s}.keras")
    print("=== Finished Loading the LSTM Model ===")
    # Make predictions.
    predicted_trajectories = step_2_predictor_training.generate_pred_trajectories(trained_lstm_models, training_trajectories, lstm_norm)
    # Extract for only the ego agent.
    training_trajectories_extracted = dict()
    for i in range(1, params.num_train + 1):
        training_trajectories_extracted[i] = training_trajectories[i][params.ego_agent]
    training_trajectories = training_trajectories_extracted
    predicted_trajectories_extracted = dict()
    for i in range(1, params.num_train + 1):
        predicted_trajectories_extracted[i] = predicted_trajectories[i][params.ego_agent]
    predicted_trajectories = predicted_trajectories_extracted

    # Calculate alphas for the indirect method.
    print("=== Calculating alphas for the indirect method ===")
    alphas_indirect = dict()
    for tau in range(params.current_time + 1, params.T + 1):
        alpha_list = []
        for i in range(1, params.num_train + 1):
            x = training_trajectories[i][tau]
            x_hat = predicted_trajectories[i][tau]
            alpha = ((x[0] - x_hat[0]) ** 2 + (x[1] - x_hat[1]) ** 2 + (x[2] - x_hat[2]) ** 2) ** (1 / 2)
            alpha_list.append(alpha)
        alphas_indirect[tau] = max(alpha_list)
    # Save the alphas.
    with open(f'alphas/{params.num_agents}-agent/STL/alphas_indirect.json', 'w') as f:
        json.dump(alphas_indirect, f)
    print("=== Finished calculating alphas for the indirect method ===")

    # Calculate alphas for the hybrid method.
    print("=== Calculating alphas for the hybrid method ===")
    alphas_hybrid = dict()
    alphas_hybrid["ground_collision"] = dict()
    alphas_hybrid["obstacle_avoidance"] = dict()
    alphas_hybrid["goal_reaching"] = dict()
    for tau in range(params.current_time + 1, params.T + 1):
        ground_collision_list = []
        obstacle_avoidance_list = []
        goal_reaching_list = []
        for i in range(1, params.num_train + 1):
            ground_trajectory = training_trajectories[i]
            predicted_trajectory = predicted_trajectories[i]
            ground_robustness_collision = evaluate_ground_collision_avoidance(ground_trajectory, tau)
            predicted_robustness_collision = evaluate_ground_collision_avoidance(predicted_trajectory, tau)
            ground_robustness_obstacle = evaluate_obstacle_avoidance(ground_trajectory, tau)
            predicted_robustness_obstacle = evaluate_obstacle_avoidance(predicted_trajectory, tau)
            ground_robustness_goal = evaluate_goal_reaching(ground_trajectory, tau)
            predicted_robustness_goal = evaluate_goal_reaching(predicted_trajectory, tau)
            ground_collision_list.append(abs(ground_robustness_collision - predicted_robustness_collision))
            obstacle_avoidance_list.append(abs(ground_robustness_obstacle - predicted_robustness_obstacle))
            goal_reaching_list.append(abs(ground_robustness_goal - predicted_robustness_goal))
        alphas_hybrid["ground_collision"][tau] = max(ground_collision_list)
        alphas_hybrid["obstacle_avoidance"][tau] = max(obstacle_avoidance_list)
        alphas_hybrid["goal_reaching"][tau] = max(goal_reaching_list)
    # Save the alphas.
    with open(f'alphas/{params.num_agents}-agent/STL/alphas_hybrid.json', 'w') as f:
        json.dump(alphas_hybrid, f)
    print("=== Finished Calculating alphas for the hybrid method ===")


if __name__ == '__main__':
    main()