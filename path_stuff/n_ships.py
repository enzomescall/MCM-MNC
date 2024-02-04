from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from path_generator import generate_path, is_within_bounds, distance
from path_tester import generate_heatmap, alter_heatmap

# set seed
np.random.seed(0)

# Generate heatmap
heatmap_shape = (50, 50)
heatmap = generate_heatmap(heatmap_shape)

start_point = (25, 25)
sonar_range = 5
sonar_falloff = 2 # distance exponent factor

def complete_search(k_generations, n_paths, start_point, desired_path_length, heatmap, sonar_range, sonar_falloff):
    # deep copy heatmap
    heatmap = heatmap.copy()

    top_path = (None, heat := 0)
    path = [start_point]
    for i in range(k_generations):
        print(f"k = {i+1}/{k_generations}")

        newpath, fullpath, newheat = generate_path(n_paths, path, desired_path_length, heatmap, sonar_range, sonar_falloff)

        print(f"Total path heat: {newheat}, total nodes visited: {len(newpath)}")

        if newheat > 0:
            path = fullpath
            heat = newheat
            top_path = (path, heat)
        else:
            print("No better path found, stopping search")
            return top_path

        alter_heatmap(heatmap, newpath, sonar_range, sonar_falloff)

    return top_path

top_path, heat = complete_search(5, 1000, start_point, 200, heatmap, sonar_range, sonar_falloff)

print(f"Top path heat: {heat}, total nodes visited: {len(top_path)}")

# Plot the heatmap and the particle's path of the top 5 paths
plt.figure(figsize=(10, 8))
plt.imshow(heatmap, interpolation='nearest')
plt.plot([y for x, y in top_path],[x for x, y in top_path],color='red', label=f'Path Heat: {heat}')
plt.scatter(start_point[1], start_point[0], color='red', marker='o', label='Start Point')
plt.legend()
plt.show()