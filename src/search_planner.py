import heapq
import json
import urllib.request
from collections import deque
from itertools import permutations

from src.data_pipeline import calculate_haversine_distance

TRAFFIC_LEVELS = ["Low", "Medium", "High", "Jam"]
TRAFFIC_MULTIPLIER = {"Low": 1.0, "Medium": 1.35, "High": 1.75, "Jam": 2.2}
AVG_SPEED_KMPH = 30.0
OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{}?steps=true&geometries=geojson&overview=false"

# Sample Mumbai road network. Road lengths are kept above the straight-line
# distance between endpoints so the Haversine heuristic stays admissible.
NODES = {
    "Lower_Parel": (18.9986, 72.8305, "Lower Parel"),
    "Worli": (19.0176, 72.8170, "Worli"),
    "Dadar": (19.0178, 72.8478, "Dadar"),
    "Bandra_West": (19.0596, 72.8295, "Bandra West"),
    "BKC": (19.0656, 72.8650, "BKC"),
    "Kurla": (19.0726, 72.8845, "Kurla"),
    "Santacruz": (19.0815, 72.8410, "Santacruz"),
    "Andheri_West": (19.1364, 72.8296, "Andheri West"),
    "Andheri_East": (19.1136, 72.8697, "Andheri East"),
    "Powai": (19.1176, 72.9060, "Powai"),
    "Ghatkopar": (19.0860, 72.9081, "Ghatkopar"),
    "Goregaon": (19.1663, 72.8526, "Goregaon"),
}

# (from, to, fallback road km, usual traffic); real street lengths replace these when OSRM answers
ROADS = [
    ("Lower_Parel", "Worli", 3.3, "Medium"),
    ("Lower_Parel", "Dadar", 3.6, "High"),
    ("Worli", "Bandra_West", 5.6, "Low"),
    ("Dadar", "Bandra_West", 6.5, "Medium"),
    ("Dadar", "Kurla", 9.4, "Jam"),
    ("Bandra_West", "BKC", 4.9, "Medium"),
    ("Bandra_West", "Santacruz", 3.5, "Medium"),
    ("BKC", "Kurla", 2.8, "High"),
    ("BKC", "Santacruz", 4.0, "Medium"),
    ("Kurla", "Ghatkopar", 3.8, "High"),
    ("Kurla", "Andheri_East", 6.3, "High"),
    ("Santacruz", "Andheri_West", 8.1, "Medium"),
    ("Santacruz", "Andheri_East", 6.1, "Medium"),
    ("Andheri_West", "Andheri_East", 6.4, "Jam"),
    ("Andheri_West", "Goregaon", 5.3, "Medium"),
    ("Andheri_East", "Goregaon", 8.0, "Low"),
    ("Andheri_East", "Powai", 5.0, "Jam"),
    ("Ghatkopar", "Powai", 4.6, "Medium"),
]


class UrbanDeliveryGraph:
    start_node = "Lower_Parel"

    def __init__(self):
        self.nodes = {k: {"lat": lat, "lon": lon, "name": name} for k, (lat, lon, name) in NODES.items()}
        self.roads = list(ROADS)
        self.shapes = {}
        self._build_edges()
        self.traffic = "Medium"
        self.risk = 0.0

    def _build_edges(self):
        self.edges = {k: [] for k in self.nodes}
        for a, b, km, traffic in self.roads:
            self.edges[a].append((b, km, traffic))
            self.edges[b].append((a, km, traffic))

    def load_street_routes(self):
        """Fetch real street paths and lengths for every link from the public OSRM server.

        Links are sent as one route (a1, b1, a2, b2, ...) so a single request covers all of
        them; every even leg is a link. If the request fails the straight lines stay.
        """
        coords = ";".join(
            f"{self.nodes[n]['lon']},{self.nodes[n]['lat']}" for a, b, _, _ in self.roads for n in (a, b)
        )
        request = urllib.request.Request(
            OSRM_URL.format(coords), headers={"User-Agent": "delivery-delay-estimator (lab project)"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as resp:
                legs = json.load(resp)["routes"][0]["legs"]
        except (OSError, ValueError, KeyError, IndexError):
            return False

        for i, (a, b, _, traffic) in enumerate(self.roads):
            leg = legs[2 * i]
            # Snapping can shave a few metres; never go below the straight line
            km = max(leg["distance"] / 1000.0, self.heuristic_distance(a, b))
            self.roads[i] = (a, b, round(km, 2), traffic)
            self.shapes[(a, b)] = [
                [round(x, 5), round(y, 5)] for step in leg["steps"] for x, y in step["geometry"]["coordinates"]
            ]
        self._build_edges()
        return True

    def shape(self, u, v):
        if (u, v) in self.shapes:
            return self.shapes[(u, v)]
        if (v, u) in self.shapes:
            return self.shapes[(v, u)][::-1]
        return [[self.nodes[n]["lon"], self.nodes[n]["lat"]] for n in (u, v)]

    def set_conditions(self, traffic="Medium", risk=0.0):
        self.traffic = traffic if traffic in TRAFFIC_LEVELS else "Medium"
        self.risk = min(max(risk, 0.0), 1.0)

    def effective_traffic(self, base_traffic):
        # Heavier city-wide traffic pushes every road up a level, lighter pulls it down
        shift = TRAFFIC_LEVELS.index(self.traffic) - 1
        level = TRAFFIC_LEVELS.index(base_traffic) + shift
        return TRAFFIC_LEVELS[min(max(level, 0), len(TRAFFIC_LEVELS) - 1)]

    def edge_cost(self, km, base_traffic):
        # Riskier orders pay more for congestion; cost never drops below km
        mult = TRAFFIC_MULTIPLIER[self.effective_traffic(base_traffic)]
        return km * (mult + self.risk * (mult - 1.0))

    def heuristic_distance(self, a, b):
        na, nb = self.nodes[a], self.nodes[b]
        return float(calculate_haversine_distance(na["lat"], na["lon"], nb["lat"], nb["lon"]))

    def _summary(self, algorithm, path, nodes_expanded):
        km = cost = minutes = 0.0
        for u, v in zip(path, path[1:]):
            dist, base = next((d, t) for n, d, t in self.edges[u] if n == v)
            km += dist
            cost += self.edge_cost(dist, base)
            minutes += dist / AVG_SPEED_KMPH * 60.0 * TRAFFIC_MULTIPLIER[self.effective_traffic(base)]
        return {
            "algorithm": algorithm,
            "path": path,
            "path_names": [self.nodes[n]["name"] for n in path],
            "coords": [pt for u, v in zip(path, path[1:]) for pt in self.shape(u, v)],
            "total_distance_km": round(km, 2),
            "estimated_time_min": round(minutes, 1),
            "path_cost": round(cost, 2),
            "nodes_expanded": nodes_expanded,
        }

    @staticmethod
    def _trace(came_from, node):
        path = [node]
        while node in came_from:
            node = came_from[node]
            path.append(node)
        return path[::-1]

    def a_star_search(self, start, goal):
        open_set = [(self.heuristic_distance(start, goal), 0.0, start)]
        g_score = {start: 0.0}
        came_from, closed = {}, set()

        while open_set:
            _, g, current = heapq.heappop(open_set)
            if current in closed:
                continue
            closed.add(current)
            if current == goal:
                return self._summary("A* search", self._trace(came_from, current), len(closed))

            for neighbor, km, traffic in self.edges[current]:
                tentative = g + self.edge_cost(km, traffic)
                if tentative < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative
                    came_from[neighbor] = current
                    f = tentative + self.heuristic_distance(neighbor, goal)
                    heapq.heappush(open_set, (f, tentative, neighbor))
        return None

    def greedy_best_first_search(self, start, goal):
        open_set = [(self.heuristic_distance(start, goal), start)]
        came_from, seen, expanded = {}, {start}, 0

        while open_set:
            _, current = heapq.heappop(open_set)
            expanded += 1
            if current == goal:
                return self._summary("Greedy best-first", self._trace(came_from, current), expanded)

            for neighbor, _, _ in self.edges[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (self.heuristic_distance(neighbor, goal), neighbor))
        return None

    def breadth_first_search(self, start, goal):
        queue = deque([start])
        came_from, seen, expanded = {}, {start}, 0

        while queue:
            current = queue.popleft()
            expanded += 1
            if current == goal:
                return self._summary("Breadth-first", self._trace(came_from, current), expanded)

            for neighbor, _, _ in self.edges[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)
        return None

    def plan_trip(self, start="Lower_Parel", drops=("Powai",), traffic="Medium", risk=0.0):
        """Visit every drop-off once, in the order with the lowest total A* cost."""
        self.set_conditions(traffic, risk)
        start = start if start in self.nodes else self.start_node
        drops = [d for d in dict.fromkeys(drops) if d in self.nodes and d != start] or ["Powai"]

        legs = {}
        def leg_cost(a, b):
            if (a, b) not in legs:
                legs[(a, b)] = self.a_star_search(a, b)["path_cost"]
            return legs[(a, b)]

        best = min(
            permutations(drops),
            key=lambda seq: sum(leg_cost(a, b) for a, b in zip((start,) + seq, seq)),
        )
        stops = (start,) + best

        results = []
        for name, search in [
            ("A* search", self.a_star_search),
            ("Greedy best-first", self.greedy_best_first_search),
            ("Breadth-first", self.breadth_first_search),
        ]:
            parts = [search(a, b) for a, b in zip(stops, stops[1:])]
            path = [start] + [node for part in parts for node in part["path"][1:]]
            results.append(self._summary(name, path, sum(p["nodes_expanded"] for p in parts)))
        return {"stops": list(stops), "results": results}

    def map_data(self):
        return {
            "nodes": [{"id": k, **v} for k, v in self.nodes.items()],
            "streets": bool(self.shapes),
            "edges": [
                {"coords": self.shape(a, b), "km": km, "traffic": self.effective_traffic(t)}
                for a, b, km, t in self.roads
            ],
        }
