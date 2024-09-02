import csv
import matplotlib.pyplot as plt
import params
import json
import numpy as np

# This function save the trajectories from csv to json with downsampling applied.
def save_trajectories(shifted = False):
    print("=== Saving Trajectories from CSV file to json file ===")
    trajectories = dict()
    if shifted:
        end_index = params.num_shifted_histories + 1
    else:
        end_index = params.num_histories + 1
    for i in range(1, end_index):
        if i % 50 == 0:
            print("Loading history indexed", i)
        trajectories[i] = dict()
        if shifted:
            file_name = f'shifted_data/history_{i}.csv'
        else:
            file_name = f'nominal_data/history_{i}.csv'
        with open(file_name, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            # Process data.
            for a in range(params.num_agents):
                trajectories[i][a] = []
            for t in range(0, len(data), params.downsample_interval):
                state = [data[t][i:i + 3] for i in range(0, len(data[t]), 3)]
                for a in range(params.num_agents):
                    trajectories[i][a].append([float(state[a][0]), float(state[a][1]), 0 - float(state[a][2])])
    # Save the trajectories.
    if shifted:
        saved_file = f'data/{params.num_agents}-agent/shifted_trajectories.json'
    else:
        saved_file = f'data/{params.num_agents}-agent/trajectories.json'
    with open(saved_file, 'w') as f:
        json.dump(trajectories, f)
    print("=== Trajectory Saving is complete. ===")
    return trajectories


# This function loads the trajectories for the processed json file.
def load_trajectories(shifted = False):
    print("=== Loading Trajectories from json file ===")
    if shifted:
        file_name = f'data/{params.num_agents}-agent/shifted_trajectories.json'
    else:
        file_name = f'data/{params.num_agents}-agent/trajectories.json'
    with open(file_name, 'r') as f:
        str_trajectories = json.load(f)
    # Convert str to integers.
    trajectories = dict()
    for history_num in str_trajectories.keys():
        trajectories[int(history_num)] = dict()
        for agent_num in str_trajectories[history_num].keys():
            trajectories[int(history_num)][int(agent_num)] = str_trajectories[history_num][agent_num]
    print("=== Trajectory Loading is complete. ===")
    return trajectories


# This function plots the trajectories of 1 history in 3d.
def plot_trajectories_3d(history, my_title, file_name):
    ax = plt.figure().add_subplot(projection='3d')
    for a in range(params.num_agents):
        x = [s[0] for s in history[a]]
        y = [s[1] for s in history[a]]
        z = [s[2] for s in history[a]]
        ax.scatter(x, y, z, label = params.legend[a], color = params.colors[a])
    ax.view_init(elev=20., azim=-35, roll=0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(my_title)
    if file_name != "":
        plt.savefig(f"plots/{params.num_agents}-agent/" + file_name + params.plotting_saving_format)
    plt.show()


def plot_trajectories_2d(history, my_title, file_name):
    # Draw the trajectories.
    for a in range(params.num_agents):
        x = [s[0] for s in history[a]]
        y = [s[1] for s in history[a]]
        plt.plot(x, y, label = params.legend[a], color = params.colors[a])
    # Draw the obstacles.
    for obstacle in params.obstacles:
        plt.plot([obstacle[0][0], obstacle[1][0]], [obstacle[0][1], obstacle[1][1]], 'bo', linestyle="--")
        plt.plot([obstacle[1][0], obstacle[3][0]], [obstacle[1][1], obstacle[3][1]], 'bo', linestyle="--")
        plt.plot([obstacle[2][0], obstacle[3][0]], [obstacle[2][1], obstacle[3][1]], 'bo', linestyle="--")
        plt.plot([obstacle[0][0], obstacle[2][0]], [obstacle[0][1], obstacle[2][1]], 'bo', linestyle="--")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title(my_title)
    if file_name != "":
        plt.savefig(f"plots/{params.num_agents}-agent/" + file_name + params.plotting_saving_format)
    plt.show()


if __name__ == '__main__':
    # First, save the trajectories. Uncomment if needed.
    # trajectories = save_trajectories()
    # shifted_trajectories = save_trajectories(shifted = True)

    # Load the trajectories.
    trajectories = load_trajectories()
    shifted_trajectories = load_trajectories(shifted = True)

    # Examine trajectories.
    reaching_distances = []
    for history in trajectories:
        for agent in range(params.num_agents):
            reaching_distances.append(trajectories[history][agent][-1][0])
    print("Mean Reaching Distance for Nominal Data:", np.average(reaching_distances))
    shifted_reaching_distances = []
    for history in shifted_trajectories:
        for agent in range(params.num_agents):
            shifted_reaching_distances.append(shifted_trajectories[history][agent][-1][0])
    print("Mean Reaching Distance for Shifted Data:", np.average(shifted_reaching_distances))

    # Plot the trajectories for nominal data.
    plot_trajectories_3d(trajectories[8], "Nominal History 8, 3D", "sample_nominal_history_8_3d")
    plot_trajectories_2d(trajectories[8], "Nominal History 8, 2D", "sample_nominal_history_8_2d")

    # Plot the trajectories for shifted data.
    plot_trajectories_3d(shifted_trajectories[8], "Shifted History 8, 3D", "sample_shifted_history_8_3d")
    plot_trajectories_2d(shifted_trajectories[8], "Shifted History 8, 2D", "sample_shifted_history_8_2d")