from numba import njit
import numpy as np
import matplotlib.pyplot as plt

# set seed
np.random.seed(0)

# Function to generate a heatmap based on an equation
def generate_heatmap(shape):
    x, y = np.meshgrid(np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[0]))
    heatmap = np.cos(5*x) + np.sin(5*y)
    return -heatmap

# Function to check if coordinates are within bounds
@njit
def is_within_bounds(x, y, heatmap):
    if x < 0 or x >= heatmap.shape[0] or y < 0 or y >= heatmap.shape[1]:
        return False
    return True

# Function to test a path
@njit
def test_path(path, heatmap, sonar_range, sonar_falloff):
    # Deep copy heatmap to avoid modifying the original
    heatmap = heatmap.copy()

    path_heat = 0
    for i in range(len(path)):
        x, y = path[i]
        path_heat += heat(x, y, heatmap, sonar_range, sonar_falloff)
    return path_heat

@njit
def heat(x, y, heatmap, sonar_range, sonar_falloff):
    heat = 0

    if not is_within_bounds(x, y, heatmap):
        print("Path out of bounds")
        return 0

    heat = heatmap[x, y]
    heatmap[x, y] = 0  # Remove heat from the visited position

    # Add the heat from the sonar
    for i in range(1, sonar_range + 1):
        for dx in range(-i, i+1):
            for dy in range(-i, i+1):
                nx = x + dx
                ny = y + dy
                if is_within_bounds(nx, ny, heatmap):
                    distance = ((dx ** 2) + (dy ** 2)) ** 0.5  # Euclidean distance
                    sonar_heat = heatmap[nx, ny] * sonar_falloff ** distance
                    heatmap[nx, ny] -= sonar_heat
                    heat += sonar_heat

    return heat

@njit
def distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

@njit
def path_length(path):
    length = 0
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        length += distance(x1, y1, x2, y2)
    return length

# Function to build a path of the right length
@njit
def append_path(path: [(float,float)], length: int, x_bound: int, y_bound: int, directions):
    current_length = 0
    i = 0

    # deep copy path
    path = path.copy()

    while current_length < length and i < len(directions):
        x, y = path[-1]
        dx, dy = directions[i]
        nx, ny = x + dx, y + dy
        if 0 <= nx < x_bound and 0 <= ny < y_bound:
            path.append((nx, ny))
            current_length += distance(x, y, nx, ny)
        i += 1

    if current_length < length:
        print("Could not reach desired length")

    return path


# Generate random directions
@njit
def generate_directions(n, step_size=1):
    directions = []
    for _ in range(n):
        dx, dy = np.random.randint(-1, 2, 2) * step_size
        directions.append((dx, dy))
    return directions


def combine_paths(a: [(float,float)], b: [(float,float)], weight: float):
    if len(a) > len(b):
        print("a is longer than b, appending b with the last points of a")
        b += [a[-1]] * (len(a) - len(b))
    elif len(b) > len(a):
        print("b is longer than a, appending a with the last points of b")
        a += [b[-1]] * (len(b) - len(a))

    combined = []
    for i in range(len(a)):
        ax, ay = a[i]
        bx, by = b[i]
        nx = (ax * weight + bx * (1 - weight))/2
        ny = (ay * weight + by * (1 - weight))/2

        # Round to nearest integer
        nx = int(nx)
        ny = int(ny)

        combined.append((nx, ny))

    return combined

@njit()
def generate_paths(n_paths, previous_path, desired_path_length, heatmap_shape, heatmap, sonar_range, sonar_falloff, direction_function=generate_directions):
    paths = []

    for _ in range(n_paths + 1):
        # Print out every 100 paths
        if _ % 100 == 0:
            print(f"Generating path {_}/{n_paths}")

        path = append_path(previous_path, desired_path_length, heatmap_shape[0], heatmap_shape[1], generate_directions(100))
        path_heat = test_path(path, heatmap, sonar_range, sonar_falloff)
        paths.append((path, path_heat))
    return paths

@njit
def complete_search(k_generations, n_paths, start_point, desired_path_length, heatmap_shape, heatmap, sonar_range, sonar_falloff):
    top_path = (None, 0)
    previous_path = [start_point]
    for i in range(k_generations):
        print(f"k = {i+1}/{k_generations}")

        paths = generate_paths(n_paths - i * 100, previous_path, desired_path_length, heatmap_shape, heatmap, sonar_range, sonar_falloff)

        # Sort paths by heat
        paths.sort(key=lambda x: x[1], reverse=True)

        # Take the top path use it to generate new paths
        previous_path = paths[0][0]

    return previous_path

# Generate heatmap
heatmap_shape = (50, 50)
heatmap = generate_heatmap(heatmap_shape)

start_point = (25, 25)
desired_path_length = 40
sonar_range = 5
sonar_falloff = 0.8

# total length = k * desired_path_length
top_path = complete_search(10, 5000, start_point, desired_path_length, heatmap_shape, heatmap, sonar_range, sonar_falloff)

top_heat = test_path(top_path, heatmap, sonar_range, sonar_falloff)

# Plot the heatmap and the particle's path of the top 5 paths
plt.figure(figsize=(10, 8))
plt.imshow(heatmap, interpolation='nearest')
plt.plot([y for x, y in top_path],[x for x, y in top_path],color='red', label=f'Path Heat: {top_heat}')
plt.scatter(start_point[1], start_point[0], color='red', marker='o', label='Start Point')
plt.legend()
plt.show()

# Test paths
"""
path = append_path([start_point], desired_path_length, heatmap_shape[0], heatmap_shape[1], generate_directions(100))
path_heat = test_path(path, heatmap, sonar_range, sonar_falloff)
print("Path heat:", path_heat)

print("Path length:", path_length(path))

path_b = append_path([start_point], desired_path_length, heatmap_shape[0], heatmap_shape[1], generate_directions(100))
path_b_heat = test_path(path_b, heatmap, sonar_range, sonar_falloff)

heat_ratio = path_heat/(path_heat + path_b_heat)

print("Path B heat:", path_b_heat)
print("Heat ratio:", heat_ratio)

combined_path = combine_paths(path, path_b, heat_ratio)
combined_path_heat = test_path(combined_path, heatmap, sonar_range, sonar_falloff)
print("Combined path heat:", combined_path_heat)

# Plot the heatmap and the particle's path
plt.figure(figsize=(10, 8))
plt.imshow(heatmap, interpolation='nearest')
plt.plot([x for x, y in path], [y for x, y in path], 'r-', label='Vessel Path')
plt.plot([x for x, y in path_b], [y for x, y in path_b], 'g-', label='Vessel Path B')
plt.plot([x for x, y in combined_path], [y for x, y in combined_path], 'b-', label='Combined Path')
plt.scatter(start_point[1], start_point[0], color='red', marker='o', label='Start Point')
plt.legend()
plt.show()
"""