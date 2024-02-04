from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from path_generator import generate_path, is_within_bounds, distance

# set seed
np.random.seed(0)

# Function to generate a heatmap based on an equation
def generate_heatmap(shape):
    x, y = np.meshgrid(np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[0]))
    heatmap = np.cos(5*x) + np.sin(5*y)

    # # make sure the whole heatmap is positive
    # min_heat = np.min(heatmap)
    # heatmap += abs(min_heat) + 1

    return -heatmap

@njit
def alter_heatmap(heatmap, path, sonar_range, sonar_falloff):
    for x,y in path:
        for i in range(sonar_range):
            for dx in range(-i, i+1):
                for dy in range(-i, i+1):
                    nx = x + dx
                    ny = y + dy
                    if is_within_bounds(nx, ny, heatmap):
                        d = distance(x, y, nx, ny)
                        if d == 0:
                            heatmap[nx, ny] = 0
                            continue
                        sonar_heat = heatmap[nx, ny] / (d ** sonar_falloff)
                        heatmap[nx, ny] -= sonar_heat

    print("Heatmap altered, sonar range:", sonar_range, "sonar falloff:", sonar_falloff)

# Generate heatmap
heatmap_shape = (50, 50)
heatmap = generate_heatmap(heatmap_shape)

start_point = (25, 25)
sonar_range = 5
sonar_falloff = 1.4 # distance exponent factor

def complete_search(k_generations, n_paths, start_point, desired_path_length, heatmap, sonar_range, sonar_falloff):
    # deep copy heatmap
    heatmap = heatmap.copy()

    top_path = (None, heat := 0)
    path = [start_point]
    for i in range(k_generations):
        print(f"k = {i+1}/{k_generations}")

        newpath, newheat = generate_path(n_paths, path, desired_path_length, heatmap, sonar_range, sonar_falloff)

        print(f"Total path heat: {newheat}, total nodes visited: {len(newpath)}")

        if newheat > 0:
            path = newpath
            heat = newheat
            top_path = (path, heat)
        else:
            print("No better path found, stopping search")
            return top_path

        alter_heatmap(heatmap, path, sonar_range, sonar_falloff)

    return top_path

top_path, heat = complete_search(10, 1000, start_point, 200, heatmap, sonar_range, sonar_falloff)

# Plot the heatmap and the particle's path of the top 5 paths
plt.figure(figsize=(10, 8))
plt.imshow(heatmap, interpolation='nearest')
plt.plot([y for x, y in top_path],[x for x, y in top_path],color='red', label=f'Path Heat: {heat}')
plt.scatter(start_point[1], start_point[0], color='red', marker='o', label='Start Point')
plt.legend()
plt.show()