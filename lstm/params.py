# Hyperparameters
num_histories = 1000 # Number of trajectories to be loaded.
num_shifted_histories = 500 # Number of shifted trajectories to be loaded.
num_agents = 5 # Total number of agents.
ego_agent = 0 # The agent that we are interested in.
T = 120 # The entire time horizon.

# Graph parameters.
if num_agents == 5:
    fixed_graph_topology = [[0, 1], [1, 2], [1, 4], [1, 3]]
elif num_agents == 7:
    fixed_graph_topology = [[0, 1], [1, 2], [1, 4], [1, 3], [1, 5], [1, 6]]
else:
    fixed_graph_topology = [[0, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9]]
distance_to_transmission_time_ratio = 0.2 # Distance * 0.2 = data transmission time.


# Obstacles.
obstacle_1 = [[56.25, 18.75], [56.25, 56.25], [18.75, 18.75], [18.75, 56.25]]
obstacle_2 = [[56.25, 93.75], [56.25, 131.25], [18.75, 93.75], [18.75, 131.25]]
obstacle_3 = [[56.25, 168.75], [56.25, 206.25], [18.75, 168.75], [18.75, 206.25]]
obstacle_4 = [[56.25, 243.75], [56.25, 281.25], [18.75, 243.75], [18.75, 281.25]]
obstacle_5 = [[131.25, 56.25], [131.25, 93.75], [93.75, 56.25], [93.75, 93.75]]
obstacle_6 = [[131.25, 131.25], [131.25, 168.75], [93.75, 131.25], [93.75, 168.75]]
obstacle_7 = [[131.25, 206.25], [131.25, 243.75], [93.75, 206.25], [93.75, 243.75]]
obstacle_8 = [[206.25, 18.75], [206.25, 56.25], [168.75, 18.75], [168.75, 56.25]]
obstacle_9 = [[206.25, 93.75], [206.25, 131.25], [168.75, 93.75], [168.75, 131.25]]
obstacle_10 = [[206.25, 168.75], [206.25, 206.25], [168.75, 168.75], [168.75, 206.25]]
obstacle_11 = [[206.25, 243.75], [206.25, 281.25], [168.75, 243.75], [168.75, 281.25]]
obstacle_12 = [[281.25, 56.25], [281.25, 93.75], [243.75, 56.25], [243.75, 93.75]]
obstacle_13 = [[281.25, 131.25], [281.25, 168.75], [243.75, 131.25], [243.75, 168.75]]
obstacle_14 = [[281.25, 206.25], [281.25, 243.75], [243.75, 206.25], [243.75, 243.75]]
obstacles = [obstacle_1, obstacle_2, obstacle_3, obstacle_4,
             obstacle_5, obstacle_6, obstacle_7, obstacle_8,
             obstacle_9, obstacle_10, obstacle_11, obstacle_12,
             obstacle_13, obstacle_14]

# Plotting hyperparameters.
if num_agents == 5:
    legend = ["Agent 1", "Agent 2", "Agent 3", "Agent 4", "Agent 5"]
    colors = ["r", "g", "b", "c", "m"]
elif num_agents == 7:
    legend = ["Agent 1", "Agent 2", "Agent 3", "Agent 4", "Agent 5", "Agent 6", "Agent 7"]
    colors = ["r", "g", "b", "c", "m", "y", "k"]
else:
    legend = ["Agent 1", "Agent 2", "Agent 3", "Agent 4", "Agent 5", "Agent 6", "Agent 7", "Agent 8", "Agent 9", "Agent 10"]
    colors = ["r", "g", "b", "c", "m", "y", "k", "orange", "purple", "brown"]
plotting_saving_format = ".pdf"

# Saving and loading hyperparameters.
downsample_interval = 100 # Take only points that are 100 timestamps separated from each other.

# Training, calibration, and test parameters.
num_train = 200
num_calibration_each_trial = 500
num_test_each_trial = 100
current_time = 50
my_seed = 12345
kde_calculation_bin_num = 200000
terminal_height = 50
communication_distance_threshold = 6
ground_height = 10
goal_location = 600
num_experiments = 50
delta = 0.2
indirect_illustration_variant_1_tau = 120

font_size = 20
label_size = 24
legend_size = 15