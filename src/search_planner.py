import heapq
from collections import deque
import numpy as np


class UrbanDeliveryGraph:
    def __init__(self):
        self.nodes = {
            "Central_Hub": {"x": 18.5204, "y": 73.8567, "name": "Central Kitchen Hub"},
            "Camp_Junction": {"x": 18.5144, "y": 73.8785, "name": "Camp Commercial Transit"},
            "Koregaon_Park": {"x": 18.5362, "y": 73.8940, "name": "Koregaon Park Restaurant Hub"},
            "Shivajinagar": {"x": 18.5314, "y": 73.8446, "name": "Shivajinagar Depot"},
            "Kothrud_Depot": {"x": 18.5074, "y": 73.8077, "name": "Kothrud Transit Waypoint"},
            "Aundh_Crossing": {"x": 18.5580, "y": 73.8075, "name": "Aundh Crossing"},
            "Baner_HighStreet": {"x": 18.5590, "y": 73.7868, "name": "Baner High Street Junction"},
            "Hinjewadi_Phase1": {"x": 18.5913, "y": 73.7389, "name": "Hinjewadi Tech Hub"},
            "Viman_Nagar": {"x": 18.5679, "y": 73.9143, "name": "Viman Nagar Drop Area"},
            "Kharadi_IT_Park": {"x": 18.5516, "y": 73.9349, "name": "Kharadi Customer Drop"},
            "Wakad_Bridge": {"x": 18.5987, "y": 73.7634, "name": "Wakad Bridge Checkpoint"},
            "Pimple_Saudagar": {"x": 18.5985, "y": 73.7997, "name": "Pimple Saudagar Customer Drop"},
        }

        self.edges = {
            "Central_Hub": [
                ("Camp_Junction", 3.2, "Medium"),
                ("Shivajinagar", 2.8, "Low"),
                ("Koregaon_Park", 4.5, "High"),
            ],
            "Camp_Junction": [
                ("Central_Hub", 3.2, "Medium"),
                ("Koregaon_Park", 2.9, "Medium"),
                ("Kharadi_IT_Park", 7.8, "Jam"),
            ],
            "Koregaon_Park": [
                ("Central_Hub", 4.5, "High"),
                ("Camp_Junction", 2.9, "Medium"),
                ("Viman_Nagar", 4.1, "Medium"),
            ],
            "Shivajinagar": [
                ("Central_Hub", 2.8, "Low"),
                ("Kothrud_Depot", 4.9, "Medium"),
                ("Aundh_Crossing", 4.6, "Low"),
            ],
            "Kothrud_Depot": [
                ("Shivajinagar", 4.9, "Medium"),
                ("Baner_HighStreet", 6.8, "High"),
            ],
            "Aundh_Crossing": [
                ("Shivajinagar", 4.6, "Low"),
                ("Baner_HighStreet", 3.1, "Low"),
                ("Pimple_Saudagar", 4.4, "Medium"),
            ],
            "Baner_HighStreet": [
                ("Aundh_Crossing", 3.1, "Low"),
                ("Kothrud_Depot", 6.8, "High"),
                ("Wakad_Bridge", 4.2, "Medium"),
                ("Hinjewadi_Phase1", 6.5, "Jam"),
            ],
            "Wakad_Bridge": [
                ("Baner_HighStreet", 4.2, "Medium"),
                ("Pimple_Saudagar", 3.8, "Low"),
                ("Hinjewadi_Phase1", 3.9, "High"),
            ],
            "Pimple_Saudagar": [
                ("Aundh_Crossing", 4.4, "Medium"),
                ("Wakad_Bridge", 3.8, "Low"),
            ],
            "Hinjewadi_Phase1": [
                ("Baner_HighStreet", 6.5, "Jam"),
                ("Wakad_Bridge", 3.9, "High"),
            ],
            "Viman_Nagar": [
                ("Koregaon_Park", 4.1, "Medium"),
                ("Kharadi_IT_Park", 3.6, "Low"),
            ],
            "Kharadi_IT_Park": [
                ("Camp_Junction", 7.8, "Jam"),
                ("Viman_Nagar", 3.6, "Low"),
            ],
        }

        self.traffic_multiplier = {
            "Low": 1.0,
            "Medium": 1.35,
            "High": 1.75,
            "Jam": 2.2,
        }

    def heuristic_distance(self, node_a, node_b):
        pos_a = self.nodes[node_a]
        pos_b = self.nodes[node_b]
        dx = (pos_a["x"] - pos_b["x"]) * 111.0
        dy = (pos_a["y"] - pos_b["y"]) * 111.0
        return np.sqrt(dx * dx + dy * dy)

    def a_star_search(self, start_node, goal_node, dynamic_traffic_override=None):
        open_set = []
        heapq.heappush(open_set, (0.0, start_node))

        came_from = {}
        g_score = {node: float("inf") for node in self.nodes}
        g_score[start_node] = 0.0

        f_score = {node: float("inf") for node in self.nodes}
        f_score[start_node] = self.heuristic_distance(start_node, goal_node)

        nodes_expanded = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            nodes_expanded += 1

            if current == goal_node:
                path = []
                curr = current
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start_node)
                path.reverse()

                total_dist = 0.0
                total_time = 0.0
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    for neighbor, dist, base_traffic in self.edges[u]:
                        if neighbor == v:
                            traffic = dynamic_traffic_override or base_traffic
                            cost_mult = self.traffic_multiplier.get(traffic, 1.0)
                            total_dist += dist
                            total_time += (dist / 30.0) * 60.0 * cost_mult

                return {
                    "algorithm": "A* Search (Informed)",
                    "path": path,
                    "path_names": [self.nodes[n]["name"] for n in path],
                    "total_distance_km": round(total_dist, 2),
                    "estimated_time_min": round(total_time, 1),
                    "nodes_expanded": nodes_expanded,
                }

            for neighbor, dist, base_traffic in self.edges.get(current, []):
                traffic = dynamic_traffic_override or base_traffic
                cost_mult = self.traffic_multiplier.get(traffic, 1.0)
                tentative_g = g_score[current] + (dist * cost_mult)

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic_distance(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def greedy_best_first_search(self, start_node, goal_node):
        open_set = []
        heapq.heappush(open_set, (self.heuristic_distance(start_node, goal_node), start_node))
        visited = set()
        came_from = {}
        nodes_expanded = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            nodes_expanded += 1

            if current == goal_node:
                path = []
                curr = current
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start_node)
                path.reverse()

                total_dist = 0.0
                total_time = 0.0
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    for neighbor, dist, traffic in self.edges[u]:
                        if neighbor == v:
                            cost_mult = self.traffic_multiplier.get(traffic, 1.0)
                            total_dist += dist
                            total_time += (dist / 30.0) * 60.0 * cost_mult

                return {
                    "algorithm": "Greedy Best-First Search",
                    "path": path,
                    "path_names": [self.nodes[n]["name"] for n in path],
                    "total_distance_km": round(total_dist, 2),
                    "estimated_time_min": round(total_time, 1),
                    "nodes_expanded": nodes_expanded,
                }

            visited.add(current)
            for neighbor, dist, _ in self.edges.get(current, []):
                if neighbor not in visited:
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (self.heuristic_distance(neighbor, goal_node), neighbor))

        return None

    def breadth_first_search(self, start_node, goal_node):
        queue = deque([(start_node, [start_node])])
        visited = {start_node}
        nodes_expanded = 0

        while queue:
            current, path = queue.popleft()
            nodes_expanded += 1

            if current == goal_node:
                total_dist = 0.0
                total_time = 0.0
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    for neighbor, dist, traffic in self.edges[u]:
                        if neighbor == v:
                            cost_mult = self.traffic_multiplier.get(traffic, 1.0)
                            total_dist += dist
                            total_time += (dist / 30.0) * 60.0 * cost_mult

                return {
                    "algorithm": "Breadth First Search (BFS)",
                    "path": path,
                    "path_names": [self.nodes[n]["name"] for n in path],
                    "total_distance_km": round(total_dist, 2),
                    "estimated_time_min": round(total_time, 1),
                    "nodes_expanded": nodes_expanded,
                }

            for neighbor, _, _ in self.edges.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def compare_all(self, start_node="Central_Hub", goal_node="Hinjewadi_Phase1", traffic_override=None):
        astar_res = self.a_star_search(start_node, goal_node, traffic_override)
        greedy_res = self.greedy_best_first_search(start_node, goal_node)
        bfs_res = self.breadth_first_search(start_node, goal_node)
        return [astar_res, greedy_res, bfs_res]
