import unittest
import step_0_data_processing
import step_1_data_analysis
import params


def detect_phi_3(history, l, t):
    centers, side_length = step_1_data_analysis.process_obstacles()
    closest_distance = float("inf")
    for center in centers:
        new_distance = max(abs(history[l][t][0] - center[0]),
                           abs(history[l][t][1] - center[1])) - side_length / 2
        closest_distance = min(closest_distance, new_distance)
    return closest_distance >= 0


def detect_phi_1_part_1(history, l):
    for t in range(0, params.T + 1):
        if history[l][t][2] <= params.ground_height or (not detect_phi_3(history, l, t)):
            return False
    return True


def detect_phi_1(history, l):
    for t in range(0, params.T + 1):
        if history[l][t][0] >= params.goal_location:
            return detect_phi_1_part_1(history, l)
    return False


def detect_somewhere(history, l, t):
    def find_neighbors(agent):
        neighbors = set()
        for edge in params.fixed_graph_topology:
            if edge[0] == agent:
                neighbors.add(edge[1])
            elif edge[1] == agent:
                neighbors.add(edge[0])
        return neighbors

    frontier = [(l, 0)]
    visited = []
    while len(frontier) > 0:
        cur = frontier.pop()
        agent, distance = cur
        visited.append(cur)
        neighbors = find_neighbors(agent)
        for next_agent in neighbors:
            next_distance = distance + params.distance_to_transmission_time_ratio * step_1_data_analysis.compute_l2_norm_3d(history[agent][t], history[next_agent][t])
            if next_distance <= params.communication_distance_threshold:
                frontier.append((next_agent, next_distance))
    visited_nodes = set([node[0] for node in visited])
    for node in visited_nodes:
        if history[node][t][2] <= params.terminal_height:
            return True
    return False


def detect_satisfaction(history):
    for t in range(0, params.T + 1):
        if not detect_somewhere(history, params.ego_agent, t):
            return False
    return detect_phi_1(history, params.ego_agent)


class TestDataAnalysis(unittest.TestCase):

    def test_data_analysis(self):
        sample_num = 100
        trajectories = step_0_data_processing.load_trajectories()
        for i in range(1, sample_num + 1):
            history = trajectories[i]
            graph = step_1_data_analysis.DynamicGraph(history)
            computed_robustness = graph.calculate_final_robustness(0, params.T)[params.ego_agent][0]
            expected_satisfaction = detect_satisfaction(history)
            if computed_robustness >= 0:
                self.assertTrue(expected_satisfaction)
            elif computed_robustness <= 0:
                self.assertFalse(expected_satisfaction)


if __name__ == '__main__':
    unittest.main()