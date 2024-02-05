from numba import njit
import numpy as np
from queue import PriorityQueue
import matplotlib.pyplot as plt

# set seed
np.random.seed(0)

@njit
def is_within_bounds(x, y, heatmap):
    if x < 0 or x >= heatmap.shape[0] or y < 0 or y >= heatmap.shape[1]:
        return False
    return True

@njit
def test_path(path, heatmap, sonar_range, sonar_falloff):
    path_heat = 0
    for x,y in path:
        path_heat += fheat(x, y, heatmap, sonar_range, sonar_falloff)
    return path_heat

@njit
def fheat(x, y, heatmap, sonar_range, sonar_falloff):
    heat = 0

    if not is_within_bounds(x, y, heatmap):
        print("Path out of bounds")
        return 0

    # Add the heat from the sonar
    for i in range(1, sonar_range + 1):
        for dx in range(-i, i+1):
            for dy in range(-i, i+1):
                nx = x + dx
                ny = y + dy
                if is_within_bounds(nx, ny, heatmap):
                    d = distance(x, y, nx, ny)
                    if d == 0:
                            heat += heatmap[nx, ny]
                            continue
                    sonar_heat = heatmap[nx, ny] / (d ** sonar_falloff)
                    heat += sonar_heat

    return heat

@njit
def distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

@njit
def path_dist(path):
    length = 0
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        length += distance(x1, y1, x2, y2)
    return length

@njit
def get_neighbors(x, y, heatmap):
    neighbors = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if is_within_bounds(nx, ny, heatmap):
                neighbors.append((nx, ny))
    return neighbors

def generate_path(n_paths, previous_path, desired_path_dist, heatmap, sonar_range, sonar_falloff):

    visited = previous_path.copy()
    start = previous_path[-1]

    path_heats = depth_first_search(start, n_paths, desired_path_dist, heatmap, sonar_range, sonar_falloff, visited)

    # return highest heat path (this heat is innacurate lol)
    sorted_path_heats = sorted(path_heats, key=lambda x: x[1], reverse=True)


    best_path, best_heat = sorted_path_heats[0]



    #print(f"Ammended path heat: {best_heat}, Path distance: {path_dist(best_path)}")

    # combinde previous path with new path
    path = previous_path + best_path[1:]

    return best_path, path, test_path(path, heatmap, sonar_range, sonar_falloff)

def depth_first_search(start, n_paths, desired_path_dist, heatmap, sonar_range, sonar_falloff, visited):
    stack = [(start, [start], fheat(start[0], start[1], heatmap, sonar_range, sonar_falloff))]
    path_heats = []
    while stack and len(path_heats) < n_paths:
        node, path, pheat = stack.pop()

        if path_dist(path) > desired_path_dist:
            path_heats.append((path, pheat))
            continue

        x, y = node
        neighbors = get_neighbors(x, y, heatmap)

        # Create a priority queue for neighbors based on heat
        pq = PriorityQueue()
        for neighbor in neighbors:
            nx, ny = neighbor
            nheat = fheat(nx, ny, heatmap, sonar_range, sonar_falloff)
            pq.put((nheat, neighbor))

        while not pq.empty():
            nheat, neighbor = pq.get()
            if (neighbor not in path):
                stack.append((neighbor, path + [neighbor], pheat+nheat))

    if not path_heats:
        print("No paths found")

    return path_heats