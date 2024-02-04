import path 
import numpy as np

# build and test out path class
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

# make path
path = path.Path(start_point, heatmap, sonar_range, sonar_falloff)


