import numpy as np
import matplotlib.pyplot as plt

# Function to generate a heatmap based on an equation
def generate_heatmap(shape):
    x, y = np.meshgrid(np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[0]))
    heatmap = np.cos(5*x) + np.sin(5*y)
    return heatmap

# Generate heatmap
heatmap_shape = (50, 50)
heatmap = generate_heatmap(heatmap_shape)

start_point = (5, 5)
desired_path_length = 50
sonar_range = 5
sonar_falloff = 0.5

# Function to check if coordinates are within bounds
def is_within_bounds(x, y, heatmap):
    if x < 0 or x >= heatmap.shape[0] or y < 0 or y >= heatmap.shape[1]:
        return False
    return True


# Function to test a path
def test_path(path, heatmap, sonar_range, sonar_falloff):
    # Deep copy heatmap to avoid modifying the original
    heatmap = heatmap.copy()

    path_heat = 0
    for i in range(len(path)):
        x, y = path[i]
        path_heat += heat(x, y, heatmap, sonar_range, sonar_falloff)
    return path_heat

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

# Generate path using stochastic gradient descent
def generate_path(heatmap, start, path_length, sonar_range, sonar_falloff):
    path = [start]
    current_heat = heat(start[0], start[1], heatmap, sonar_range, sonar_falloff)
    for i in range(path_length):
        x, y = path[-1]
        best_heat = current_heat
        best_move = (0, 0)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not is_within_bounds(nx, ny, heatmap):
                    continue
                new_heat = heat(nx, ny, heatmap, sonar_range, sonar_falloff)
                if new_heat > best_heat:
                    best_heat = new_heat
                    best_move = (dx, dy)
        if best_move == (0, 0):
            print("Stuck at", x, y)
            break
        path.append((x + best_move[0], y + best_move[1]))
        current_heat = best_heat
    return path

# Test path
path = [(i, i) for i in range(desired_path_length)]
path_heat = test_path(path, heatmap, sonar_range, sonar_falloff)
print("Path heat:", path_heat)

# Plot the heatmap and the particle's path
plt.figure(figsize=(10, 8))
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.plot([x for x, y in path], [y for x, y in path], 'r-', label='Particle Path')
plt.scatter(start_point[1], start_point[0], color='red', marker='o', label='Start Point')
plt.legend()
plt.show()