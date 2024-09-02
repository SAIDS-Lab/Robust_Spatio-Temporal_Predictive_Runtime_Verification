import params
import step_0_data_processing
import matplotlib.pyplot as plt


# Define a graph data structure.
class DynamicGraph:

    def __init__(self, history):
        # Process nodes.
        self.nodes = list(history.keys())
        self.nodes.sort()
        # Process states.
        self.states = []
        for t in range(len(history[0])):
            cur_state = dict()
            for a in self.nodes:
                cur_state[a] = history[a][t]
            self.states.append(cur_state)
        # Process edges.
        self.edges = []
        for t in range(len(self.states)):
            cur_edges = []
            for edge in params.fixed_graph_topology:
                cur_edges.append([edge[0], edge[1], params.distance_to_transmission_time_ratio * compute_l2_norm_3d(self.states[t][edge[0]], self.states[t][edge[1]])])
            self.edges.append(cur_edges)

    def find_neighbor_nodes(self, node, t):
        neighbors = []
        for edge in self.edges[t]:
            if edge[0] == node:
                neighbors.append((edge[1], edge[2]))
            elif edge[1] == node:
                neighbors.append((edge[0], edge[2]))
        return neighbors  # Return a list of tuples with (neighbor, weight).

    def find_minimum_distance_on_graph(self, node_1, node_2, t):
        # Run dijkstra's algorithm.
        distance = dict()
        unvisited = self.nodes.copy()
        for node in self.nodes:
            distance[node] = float("inf")
        distance[node_1] = 0
        while len(unvisited) > 0:
            # Find the node with the minimum distance.
            min_distance = float("inf")
            min_node = None
            for node in unvisited:
                if distance[node] < min_distance:
                    min_distance = distance[node]
                    min_node = node
            unvisited.remove(min_node)
            neighbors = self.find_neighbor_nodes(min_node, t)
            for (neighbor, weight) in neighbors:
                distance[neighbor] = min(distance[neighbor], distance[min_node] + weight)
        return distance[node_2]

    def find_minimum_distance_matrix(self, t):
        distance_matrix = dict()
        for node_1 in self.nodes:
            distance_matrix[node_1] = dict()
            for node_2 in self.nodes:
                distance_matrix[node_1][node_2] = self.find_minimum_distance_on_graph(node_1, node_2, t)
        return distance_matrix

    """
    From here on, we design offline monitor functions.
    """
    def calculate_predicate_robustness(self, predicate_function):
        # Monitor for predicates.
        robustness_matrix = dict()
        for l in self.nodes:
            robustness_matrix[l] = dict()
            for tau0 in range(len(self.states)):
                robustness_matrix[l][tau0] = predicate_function(l, tau0)
        return robustness_matrix

    def calculate_negation_robustness(self, robustness_matrix):
        # Monitor for negation.
        negation_robustness_matrix = dict()
        for l in self.nodes:
            negation_robustness_matrix[l] = dict()
            for tau0 in range(len(self.states)):
                negation_robustness_matrix[l][tau0] = 0 - robustness_matrix[l][tau0]
        return negation_robustness_matrix

    def calculate_conjunction_robustness(self, robustness_matrix_1, robustness_matrix_2):
        # Monitor for conjunction.
        conjunction_robustness_matrix = dict()
        for l in self.nodes:
            conjunction_robustness_matrix[l] = dict()
            for tau0 in range(len(self.states)):
                conjunction_robustness_matrix[l][tau0] = min(robustness_matrix_1[l][tau0], robustness_matrix_2[l][tau0])
        return conjunction_robustness_matrix

    def calculate_disjunction_robustness(self, robustness_matrix_1, robustness_matrix_2):
        # Monitor for disjunction.
        disjunction_robustness_matrix = dict()
        for l in self.nodes:
            disjunction_robustness_matrix[l] = dict()
            for tau0 in range(len(self.states)):
                disjunction_robustness_matrix[l][tau0] = max(robustness_matrix_1[l][tau0], robustness_matrix_2[l][tau0])
        return disjunction_robustness_matrix

    def calculate_always_robustness(self, robustness_matrix, t1, t2):
        # Monitor for always.
        always_robustness_matrix = dict()
        for l in self.nodes:
            always_robustness_matrix[l] = dict()
            for tau0 in range(len(self.states)):
                if tau0 + t2 >= len(self.states):
                    always_robustness_matrix[l][tau0] = 0 - float("inf")
                else:
                    always_robustness_matrix[l][tau0] = min([robustness_matrix[l][tau] for tau in range(tau0 + t1, tau0 + t2 + 1)])
        return always_robustness_matrix

    def calculate_eventually_robustness(self, robustness_matrix, t1, t2):
        # Monitor for eventually.
        eventually_robustness_matrix = dict()
        for l in self.nodes:
            eventually_robustness_matrix[l] = dict()
            for tau0 in range(len(self.states)):
                if tau0 + t2 >= len(self.states):
                    eventually_robustness_matrix[l][tau0] = 0 - float("inf")
                else:
                    eventually_robustness_matrix[l][tau0] = max([robustness_matrix[l][tau] for tau in range(tau0 + t1, tau0 + t2 + 1)])
        return eventually_robustness_matrix

    def calculate_reach_robustness(self, robustness_matrix_1, robustness_matrix_2, d_1, d_2):
        # First invert the robustness matrices.
        inverted_robustness_matrix_1 = invert_matrix(robustness_matrix_1)
        inverted_robustness_matrix_2 = invert_matrix(robustness_matrix_2)
        # Now, process for each time.
        inverted_robustness_matrix = dict()
        for tau0 in range(len(self.states)):
            inverted_robustness_matrix[tau0] = self.__monitor_reach_instantaneous(tau0, inverted_robustness_matrix_1[tau0], inverted_robustness_matrix_2[tau0], d_1, d_2)
        # Invert the inverted robustness matrix.
        reach_robustness_matrix = invert_matrix(inverted_robustness_matrix)
        return reach_robustness_matrix

    def calculate_escape_robustness(self, robustness_matrix, d_1, d_2):
        # First invert the robustness matrix.
        inverted_robustness_matrix = invert_matrix(robustness_matrix)
        # Now, process for each time.
        inverted_escape_robustness_matrix = dict()
        for tau0 in range(len(self.states)):
            inverted_escape_robustness_matrix[tau0] = self.__monitor_escape_instantaneous(tau0, inverted_robustness_matrix[tau0], d_1, d_2)
        # Invert the inverted escape robustness matrix.
        escape_robustness_matrix = invert_matrix(inverted_escape_robustness_matrix)
        return escape_robustness_matrix

    def calculate_somewhere_robustness(self, robustness_matrix, d_1, d_2):
        # Initialize an all true matix.
        all_true_matrix = dict()
        for l in self.nodes:
            all_true_matrix[l] = dict()
            for tau0 in range(len(self.states)):
                all_true_matrix[l][tau0] = float("inf")
        return self.calculate_reach_robustness(all_true_matrix, robustness_matrix, d_1, d_2)

    def calculate_everywhere_robustness(self, robustness_matrix, d_1, d_2):
        # Negate the robustness matrix.
        negated_robustness_matrix = self.calculate_negation_robustness(robustness_matrix)
        # Calculate the somewhere robustness.
        somewhere_negated_robustness_matrix = self.calculate_somewhere_robustness(negated_robustness_matrix, d_1, d_2)
        # Negate the somewhere robustness.
        return self.calculate_negation_robustness(somewhere_negated_robustness_matrix)

    def calculate_surround_robustness(self, robustness_matrix_1, robustness_matrix_2, d):
        # Calculate the negation disjunction matrix.
        negation_disjunction_matrix = self.calculate_negation_robustness(self.calculate_disjunction_robustness(robustness_matrix_1, robustness_matrix_2))
        # Calculate the reach matrix.
        reach_matrix = self.calculate_reach_robustness(robustness_matrix_1, negation_disjunction_matrix, 0, d)
        # Calculate the negation reach matrix.
        negation_reach_matrix = self.calculate_negation_robustness(reach_matrix)
        # Calculate the escape matrix.
        escape_matrix = self.calculate_escape_robustness(robustness_matrix_1, d, float("inf"))
        # Calculate the negation escape matrix.
        negation_escape_matrix = self.calculate_negation_robustness(escape_matrix)
        # Calculate the conjunction matrix.
        return self.calculate_conjunction_robustness(self.calculate_conjunction_robustness(robustness_matrix_1, negation_reach_matrix), negation_escape_matrix)

    """
    Define some useful predicate functions.
    """
    def closest_euclidean_distance_to_obstacles(self, l, t):
        centers, side_length = process_obstacles()
        closest_distance = float("inf")
        for center in centers:
            new_distance = max(abs(self.states[t][l][0] - center[0]),
                               abs(self.states[t][l][1] - center[1])) - side_length / 2
            closest_distance = min(closest_distance, new_distance)
        return closest_distance

    def goal_reaching(self, l, t):
        return self.states[t][l][0] - params.goal_location

    def ground_collision_avoidance_check(self, l, t):
        return self.states[t][l][2] - params.ground_height

    def communication_to_terminal_check(self, l, t):
        return params.terminal_height - self.states[t][l][2]

    """
    Define some useful specification functions.
    """
    def calculate_success_reach_robustness(self, t1, t2):
        # Globally, between tau0 + t1 and tau0 + t2, (there should be no collision with obstacles and with the ground).
        # And eventually between tau0 + t1 and tau0 + t2, the agent reaches the goal.

        # First, calculate predicate matrix for obstacle_avoidance.
        predicate_matrix_obstacle_avoidance = self.calculate_predicate_robustness(self.closest_euclidean_distance_to_obstacles)
        # Then, calculate the predicate matrix for the ground_collision_avoidance.
        predicate_matrix_collision_avoidance = self.calculate_predicate_robustness(self.ground_collision_avoidance_check)
        # Calculate the conjunction matrix.
        conjunction_matrix_obstacle_ground_avoidance = self.calculate_conjunction_robustness(predicate_matrix_obstacle_avoidance,predicate_matrix_collision_avoidance)
        # Calculate the always matrix.
        always_conjunction_matrix_obstacle_ground_avoidance = self.calculate_always_robustness(conjunction_matrix_obstacle_ground_avoidance, t1, t2)
        # Calculate the predicate matrix for goal reaching.
        predicate_matrix_goal_reaching = self.calculate_predicate_robustness(self.goal_reaching)
        # Calculate the eventually matrix.
        eventually_matrix_goal_reaching = self.calculate_eventually_robustness(predicate_matrix_goal_reaching, t1, t2)
        # Calculate the final matrix.
        final_robustness_matrix = self.calculate_conjunction_robustness(always_conjunction_matrix_obstacle_ground_avoidance, eventually_matrix_goal_reaching)
        return final_robustness_matrix

    def calculate_maintain_connection_terminal(self, t1, t2):
        # Globally, between tau0 + t1 and tau0 + t2, there should exists an agent at most distance_threshold away from the ego agent.
        # that remains a height below the allowed communication height.

        # First, calculate the communication_to_terminal_check predicate matrix.
        predicate_matrix_communication_to_terminal_check = self.calculate_predicate_robustness(self.communication_to_terminal_check)
        # Calculate the somewhere matrix.
        somewhere_matrix_communication_to_terminal_check = self.calculate_somewhere_robustness(predicate_matrix_communication_to_terminal_check, 0, params.communication_distance_threshold)
        # Calculate the always matrix.
        final_matrix = self.calculate_always_robustness(somewhere_matrix_communication_to_terminal_check, t1, t2)
        return final_matrix

    def calculate_final_robustness(self, t1, t2):
        success_reach_matrix = self.calculate_success_reach_robustness(t1, t2)
        maintain_connection_matrix = self.calculate_maintain_connection_terminal(t1, t2)
        return self.calculate_conjunction_robustness(success_reach_matrix, maintain_connection_matrix)

    """
    Define some helper methods.
    """
    def __monitor_bounded_reach_instantaneous(self, tau0, s_1, s_2, d_1, d_2):
        s = dict()
        # Initialize s.
        for node in self.nodes:
            if d_1 == 0:
                s[node] = s_2[node]
            else:
                s[node] = 0 - float("inf")
        # Initialize Q.
        Q = []
        for node in self.nodes:
            Q.append((node, s_2[node], 0))
        # Start the iteration.
        while len(Q) > 0:
            Q_p = []
            for (l, v, d) in Q:
                # Find the connected nodes.
                neighbors = self.find_neighbor_nodes(l, tau0)
                for (l_p, w) in neighbors:
                    v_p = min(v, s_1[l_p])
                    d_p = d + w
                    if d_1 <= d_p <= d_2:
                        s[l_p] = max(s[l_p], v_p)
                    if d_p < d_2:
                        # Search for existence.
                        flag = False
                        for i in range(len(Q_p)):
                            element = Q_p[i]
                            if element[0] == l_p and element[2] == d_p:
                                flag = True
                                Q_p.pop(i)
                                Q_p.append((l_p, max(v_p, element[1]), d_p))
                                break
                        if not flag:
                            Q_p.append((l_p, v_p, d_p))
            Q = Q_p
        return s

    def __monitor_unbounded_reach_instantaneous(self, tau0, s_1, s_2, d_1):
        if d_1 == 0:
            return s_2
        else:
            max_edge_weight = -float("inf")
            for edge in self.edges[t]:
                max_edge_weight = max(max_edge_weight, edge[2])
            if max_edge_weight == -float("inf"):
                raise ValueError("The graph is empty.")
            s = self.__monitor_bounded_reach_instantaneous(tau0, s_1, s_2, d_1, d_1 + max_edge_weight)
            T = self.nodes.copy()
            while len(T) > 0:
                T_p = []
                for l in T:
                    neighbors = self.find_neighbor_nodes(l, tau0)
                    for (l_p, w) in neighbors:
                        v_p = max(min(s[l], s_1[l_p]), s[l_p])
                        if v_p != s[l_p]:
                            s[l_p] = v_p
                            T_p.append(l_p)
                T = T_p
            return s

    def __monitor_reach_instantaneous(self, tau0, s_1, s_2, d_1, d_2):
        if d_2 != float("inf"):
            return self.__monitor_bounded_reach_instantaneous(tau0, s_1, s_2, d_1, d_2)
        else:
            return self.__monitor_unbounded_reach_instantaneous(tau0, s_1, s_2, d_1)

    def __monitor_escape_instantaneous(self, tau0, s_1, d_1, d_2):
        D = self.find_minimum_distance_matrix(tau0)
        e = dict()
        for l in self.nodes:
            e[l] = dict()
            for l_p in self.nodes:
                if l == l_p:
                    e[l][l_p] = s_1[l]
                else:
                    e[l][l_p] = 0 - float("inf")
        T = [(l, l) for l in self.nodes]
        while len(T) > 0:
            e_p = e.copy()
            T_p = []
            for (l_1, l_2) in T:
                for (l_1_p, w) in self.find_neighbor_nodes(l_1, tau0):
                    v = max(e[l_1_p][l_2], min(s_1[l_1_p], e[l_1][l_2]))
                    if v != e[l_1_p][l_2]:
                        T_p.append((l_1_p, l_2))
                        e_p[l_1_p][l_2] = v
            T = T_p.copy()
            e = e_p.copy()
        s = dict()
        for l in self.nodes:
            s[l] = 0 - float("inf")
            for l_p in self.nodes:
                if d_1 <= D[l][l_p] <= d_2:
                    s[l] = max(s[l], e[l][l_p])
        return s


"""
The following are some util functions.
"""
def compute_l2_norm_2d(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def compute_l2_norm_3d(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5


# Check if an obstacle is square.
def is_square(obstacle):
    assert len(obstacle) == 4
    # Check if the lengths of the sides are the same.
    length_1 = compute_l2_norm_2d(obstacle[0], obstacle[1])
    length_2 = compute_l2_norm_2d(obstacle[2], obstacle[3])
    length_3 = compute_l2_norm_2d(obstacle[0], obstacle[2])
    length_4 = compute_l2_norm_2d(obstacle[1], obstacle[3])
    assert length_1 == length_2 == length_3 == length_4
    # Check if the input is in the right order.
    assert obstacle[0][0] == obstacle[1][0] and obstacle[0][1] <= obstacle[1][1]
    assert obstacle[0][0] >= obstacle[2][0] and obstacle[0][1] == obstacle[2][1]
    assert obstacle[1][0] >= obstacle[3][0] and obstacle[1][1] == obstacle[3][1]
    assert obstacle[2][0] == obstacle[3][0] and obstacle[2][1] <= obstacle[3][1]


def invert_matrix(matrix):
    new_matrix = dict()
    index = list(matrix.keys())[0]
    for y in matrix[index].keys():
        new_matrix[y] = dict()
        for x in matrix.keys():
            new_matrix[y][x] = matrix[x][y]
    return new_matrix


# this function takes in the obstacle information (with vertices) and process them into
# the format of [[center_1, center_2, ...], side_length]
def process_obstacles():
    centers = []
    side_lengths = []
    for obstacle in params.obstacles:
        # First check if the obstacle is indeed a square from view above.
        is_square(obstacle)
        # Compute the center of the square.
        center = [0.5 * (obstacle[0][0] + obstacle[2][0]), 0.5 * (obstacle[0][1] + obstacle[1][1])]
        length = compute_l2_norm_2d(obstacle[0], obstacle[1])
        centers.append(center)
        side_lengths.append(length)
    assert len(set(side_lengths)) == 1
    return [centers, side_lengths[0]]


if __name__ == '__main__':
    trajectories = step_0_data_processing.load_trajectories()
    shifted_trajectories = step_0_data_processing.load_trajectories(shifted = True)
    # Define some analysis hyperparameters.

    num_analysis = 100
    # Perform analysis on the nominal data.
    robustnesses_success_reach = []
    robustnesses_connection = []
    robustnesses_final = []
    print("Analyze for the nominal data.")
    for i in range(1, num_analysis + 1):
        if i % 50 == 0:
            print("Performing analysis for history indexed at", i)
        history = trajectories[i]
        graph = DynamicGraph(history)

        # Analyze successful reach.
        robustness_success_reach = graph.calculate_success_reach_robustness(0, params.T)[params.ego_agent][0]
        robustnesses_success_reach.append(robustness_success_reach)

        # Analyze connection maintenance.
        robustness_connection = graph.calculate_maintain_connection_terminal(0, params.T)[params.ego_agent][0]
        robustnesses_connection.append(robustness_connection)

        # Analyze the final robustness.
        robustness_final = graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
        robustnesses_final.append(robustness_final)

    # Perform analysis on the shifted data.
    shifted_robustnesses_success_reach = []
    shifted_robustnesses_connection = []
    shifted_robustnesses_final = []
    print("Analyze for the shifted data.")
    for i in range(1, num_analysis + 1):
        if i % 50 == 0:
            print("Performing analysis for history indexed at", i)
        history = shifted_trajectories[i]
        graph = DynamicGraph(history)

        # Analyze successful reach.
        shifted_robustness_success_reach = graph.calculate_success_reach_robustness(0, params.T)[params.ego_agent][0]
        shifted_robustnesses_success_reach.append(shifted_robustness_success_reach)

        # Analyze connection maintenance.
        shifted_robustness_connection = graph.calculate_maintain_connection_terminal(0, params.T)[params.ego_agent][0]
        shifted_robustnesses_connection.append(shifted_robustness_connection)

        # Analyze the final robustness.
        shifted_robustness_final = graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
        shifted_robustnesses_final.append(shifted_robustness_final)

    # Let's make some plots
    # First plot the robustness of the successful reach over the trajectories.
    plt.hist(robustnesses_success_reach, bins=100, label = "Nominal")
    plt.hist(shifted_robustnesses_success_reach, bins = 100, label = "Shifted")
    plt.title("Robustness for Successfully Reaching the Goal Location with No Collision")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"plots/{params.num_agents}-agent/robustnesses_success_reach" + params.plotting_saving_format)
    plt.show()

    # Second, plot the robustness of the connection maintenance over the trajectories.
    plt.hist(robustnesses_connection, bins=100, label = "Nominal")
    plt.hist(shifted_robustnesses_connection, bins = 100, label = "Shifted")
    plt.title("Robustness for Maintaining Connection with Terminal")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"plots/{params.num_agents}-agent/robustnesses_connection" + params.plotting_saving_format)
    plt.show()

    # Third, plot the robustness for the final specification over the trajectories.
    plt.hist(robustnesses_final, bins=100, label = "Nominal")
    plt.hist(shifted_robustnesses_final, bins = 100, label = "Shifted")
    plt.title("Robustness for the Final Specification")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"plots/{params.num_agents}-agent/robustnesses_final" + params.plotting_saving_format)
    plt.show()