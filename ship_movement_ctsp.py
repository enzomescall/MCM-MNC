import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# Function to generate a heatmap based on an equation
def generate_heatmap(shape):
    x, y = np.meshgrid(np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[0]))
    heatmap = - x**2 - y**2
    heatmap = np.cos(5*x) + np.sin(5*y)
    return heatmap

# Function to perform Fast Marching Method for CTSP
def fast_marching_ctsp(heatmap, start, path_length):
    # Calculate the distance transform on the inverted heatmap
    inverted_heatmap = np.max(heatmap) - heatmap
    distance_transform = distance_transform_edt(inverted_heatmap)

    # Initialize the starting point
    i, j = start
    path = [start]
    current_length = 0

    # Iterate until the path reaches the desired length
    while current_length < path_length:
        neighbors = []
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = i + di, j + dj
                if 0 <= ni < heatmap.shape[0] and 0 <= nj < heatmap.shape[1]:
                    neighbors.append((ni, nj))

        # Find the neighbor with the lowest cost in the distance transform
        costs = [distance_transform[ni, nj] for ni, nj in neighbors]
        # Here we can implement a more effective search algorithm
        min_index = np.argmin(costs)
        ni, nj = neighbors[min_index]

        # Update the current position and length
        i, j = ni, nj
        path.append((i, j))
        current_length += 1

        # Update the distance transform to avoid revisiting the same point
        distance_transform[i, j] = np.inf

    return np.array(path)

# Main program
heatmap_shape = (50, 50)
heatmap = generate_heatmap(heatmap_shape)

# Starting point for the particle
start_point = (5, 5)

# Path length for the particle
desired_path_length = 1000

# Use Fast Marching Method for CTSP
path = fast_marching_ctsp(heatmap, start_point, desired_path_length)

# Plot the heatmap and the particle's path
plt.figure(figsize=(10, 8))

# Plot the heatmap
plt.imshow(heatmap, cmap='viridis', extent=(-1, 1, -1, 1))
plt.colorbar(label='Heatmap Value')

# Plot the particle's path
plt.plot(path[:, 1] / (heatmap_shape[1] - 1) * 2 - 1, path[:, 0] / (heatmap_shape[0] - 1) * 2 - 1, 'r-', linewidth=2)
plt.scatter(start_point[1] / (heatmap_shape[1] - 1) * 2 - 1, start_point[0] / (heatmap_shape[0] - 1) * 2 - 1, color='red', marker='o', label='Start')

plt.title('Search Path of Host Vessel for Submersible')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
