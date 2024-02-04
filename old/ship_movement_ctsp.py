import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from numba import jit, njit


# set seed
np.random.seed(0)

# Function to generate a heatmap based on an equation
def generate_heatmap(shape):
    x, y = np.meshgrid(np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[0]))
    noise = 0.05 * np.random.normal(size=x.shape)  # Random noise
    gauss_peak = np.exp(-(x**2 + y**2) / 10)  # Gaussian peak at the center
    sine_pattern = np.sin(5 * x) + np.cos(5 * y)  # Sinusoidal pattern

    heatmap = 0.7 * gauss_peak + 0.2 * sine_pattern + 0.1 + noise
    return heatmap
########################################
# Useful functions that are always run #
########################################

@njit
def is_within_bounds(x, y, heatmap):
    if x < 0 or x >= heatmap.shape[0] or y < 0 or y >= heatmap.shape[1]:
        return False
    return True


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

@njit
def distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def path_length(path):
    length = 0
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        length += distance(x1, y1, x2, y2)
    return length

@njit
def get_neighbors(x, y, heatmap, distance=1):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx = x + dx*distance
            ny = y + dy*distance
            if is_within_bounds(nx, ny, heatmap):
                neighbors.append(((nx, ny), heatmap[nx, ny]))
    return neighbors

def remove_heat(x, y, heatmap, sonar_range, sonar_falloff):
    if not is_within_bounds(x, y, heatmap):
        print("Path out of bounds")
        return 0
    
    heatmap[x, y] = 0  # Remove heat from the visited position

    # remove the heat from the sonar
    for i in range(1, sonar_range + 1):
        visited = []
        visited.append(((x, y), heatmap[x, y]))
        for dx in range(-i, i+1):
            for dy in range(-i, i+1):
                nx = x + dx
                ny = y + dy
                if is_within_bounds(nx, ny, heatmap) and (nx, ny) not in visited:
                    sonar_heat = heatmap[nx, ny] / (sonar_falloff * distance(x, y, nx, ny) + 1)
                    visited.append(((nx, ny), sonar_heat))
                    heatmap[nx][ny] -= sonar_heat

# Function to attempt to perform Fast Marching Method for CTSP

def fast_marching_ctsp(heatmap, start, max_length, sonar_range, sonar_falloff, distance_damper, max_iter=100):
    # Deep copy heatmap
    heatmap = heatmap.copy()
    # Initialize the starting point
    x, y = start
    path = [start]
    heatmap[start] = 0
    current_length = 0
    i = 0
    # Iterate until the path reaches the desired length
    while current_length < max_length and i < max_iter:
        # notify on every 10 iterations
        if i % 10 == 0:
            print(f"Current length: {current_length}, Iteration: {i}")
        
        # Get the neighbors of the current point
        neighbors = get_neighbors(x, y, heatmap)

        # # Find the best point on the board
        # bestx, besty = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        # print(f"Best: {bestx, besty}, best heat: {heatmap[bestx][besty]}")

        # Find the best point within 5 sonar distances
        sonarx, sonary = 0, 0
        for j in range(1, 5*sonar_range + 1):
            visited = []
            visited.append(((x, y), heatmap[x, y]))
            for dx in range(-j, j+1):
                for dy in range(-j, j+1):
                    nx = x + dx
                    ny = y + dy
                    if is_within_bounds(nx, ny, heatmap) and (nx, ny) not in visited and heatmap[nx, ny] > heatmap[sonarx, sonary]:
                            sonarx, sonary = nx, ny
        
        print(f"Sonar: {sonarx, sonary}, sonar heat: {heatmap[sonarx][sonary]}")

        max_score = -np.inf
        for nx, ny in neighbors:
            print(f"Neighbor: {nx, ny}")
            curr_heat = heatmap[nx][ny]
            score = curr_heat - distance_damper(distance(x, y, sonarx, sonary))
            print(f"Heat: {curr_heat}, score: {score}, distance: {distance(x, y, sonarx, sonary)}, and distance damper: {distance_damper(distance(x, y, sonarx, sonary))}")
            if score > max_score:
                print("Updating max score")
                max_score = score
                x, y = nx, ny

        # Remove the heat from the previously visited position
        remove_heat(x, y, heatmap, sonar_range, sonar_falloff)

        # Add the best neighbor to the path
        path.append((x, y))

        # Update the current length
        current_length += distance(*path[-2], *path[-1])

        i += 1

    return path, heatmap

def avoid_sonar_search(heatmap, start, max_length, sonar_range, distance_damper, max_iter=100):
    path = [start]
    heatmap = heatmap.copy()

    heatmap_skip = int(sonar_range)
    
    current_length = 0
    i = 0
    while current_length < max_length and i < max_iter:
        if i % 10 == 0:
            print(f"Current length: {current_length}, Iteration: {i}")

        x, y = path[-1]
        neighbors = get_neighbors(x, y, heatmap, heatmap_skip)

        # find best point in heatmap not in path
        best = heatmap.copy()
        bestx, besty = np.unravel_index(np.argmax(best), best.shape)
        while (bestx, besty) in path:
            best[bestx, besty] = -np.inf
            bestx, besty = np.unravel_index(np.argmax(best), best.shape)

        max_score = -np.inf
        for point, heat in neighbors:
            if point in path:
                score = -distance_damper(distance(point[0], point[1], bestx, besty))
            else:
                score = heat - distance_damper(distance(point[0], point[1], bestx, besty))
            if score > max_score:
                max_score = score
                x, y = point

        path.append((x, y))
        current_length += distance(*path[-2], *path[-1])

        if distance(*path[-2], *path[-1]) < 1:
            print("Path stuck, diagnosing")
            print(f"Current length: {current_length}, Iteration: {i}")
            print(f"Current path: {path}")
            print(f"Current heat: {heatmap[x][y]}")
            print(f"Neghbors: {neighbors}")
            print(f"Best point: {bestx, besty}")
            print(f"max score: {max_score}")

            return path

        i += 1

    return path


# Main program
print("Running main program")
heatmap_shape = (100, 100)
heatmap = generate_heatmap(heatmap_shape)

starting_point = (50, 50)
sonar_range = 1
sonar_falloff = 3
max_length = 500

@njit
def distance_damper(distance):
    return np.log(distance + 1)

print("Generating path")
path = avoid_sonar_search(heatmap, starting_point, max_length, sonar_range, distance_damper, max_length)


# path, removed_heatmap = fast_marching_ctsp(heatmap, starting_point, max_length, sonar_range, sonar_falloff, distance_damper, 500)

print("Plotting")

# Plot the heatmap and the path
plt.imshow(heatmap, interpolation='nearest')
plt.plot([y for x, y in path], [x for x, y in path], color='red', label=f'Path 2 Heat: {test_path(path, heatmap, 5, 0.5)}')

plt.show()

import matplotlib.animation as animation

fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.imshow(heatmap, interpolation='nearest')
    ax.plot([y for x, y in path[:frame]], [x for x, y in path[:frame]], color='red', label=f'Path 2 Heat: {test_path(path[:frame], heatmap, 5, 0.5)}')
    ax.set_title(f'Frame {frame}')
    ax.legend()

ani = animation.FuncAnimation(fig, update, frames=len(path), interval=200)
plt.show()

