from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from buoyancy import position_to_latlong
import sys
from euler_maryama import euler_maryama, get_wiener_deltas
sys.path.append('ADE-python/structured_grid')

from path_tester import check_search


def main(paths, sonar_range, ballast, initial_position):
    # get positions from euler_maryama
    # isolate one sumbersible path
    # get search path from path_tester
    # run search on heatmap from euler_maryama
    # if submersible path is ever within sonar, declare found
    # end process after certain amount of time?

    # so the idea is that as we run euler_maryama,
    # we can also run the search path every n iterations

    dt = 0.1 # time step
    T = 10 # total time
    n = int(T / dt) # number of time steps
    damper = 2 # multiplier for omega
    mass = 11800
    lat_long = (37.5, 21.5)

    start_position = (0, 0, -2000)
    start_velocity = (-10, 10, 0.1)

    # print('Values have been initialized')
    # print(f'dt: {dt}, T: {T}, n: {n}, mass: {mass}')
    # print(f'Starting position: {start_position}, starting velocity: {start_velocity}')

    # # Euler-Maruyama method
    # print('Starting Euler-Maruyama method...')

    num_paths = 1000

    timestep = []

    for i in range(n):
        all_x = []
        all_y = []

        for path in paths:
            all_x = np.append(all_x, path[i, 0])
            all_y = np.append(all_y, path[i, 1])

        timestep.append((all_x, all_y))

    heatmaps = [] # heatmaps for each time step
    for t, _ in enumerate(timestep):
        all_x = timestep[t][0]
        all_y = timestep[t][1]

        heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=25, density=True)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        heatmap = gaussian_filter(heatmap, sigma=1)
        heatmaps.append(heatmap)

    # Complete search needs (k_generations, n_paths, start_point,
    #                        desired_path_length, heatmap,
    #                        sonar_range, sonar_falloff)
    
    k_generations = 1
    n_paths = 100
    start_point = initial_position
    print(f'Starting point of vessel: {start_point}')
    desired_path_length = 25
    sonar_falloff = 2

    subs_found, time = check_search(k_generations, n_paths, start_point, desired_path_length, heatmaps, sonar_range, sonar_falloff, paths, heatmap_skip = 1)

    print(f"Submersibles found: {subs_found} out of {num_paths}")
    # log to file how many subs found and the parameters used
    with open("search_results.txt", "a") as f:
        f.write(f"Submersibles found: {subs_found} out of {num_paths}, Sonar range: {sonar_range}, Ballast: {ballast}, Initial position: {initial_position}\n")
        if time:
            f.write(f"Time to find all subs: {time}\n")

if __name__ == "__main__":
    with open("search_results.txt", "a") as f:
        f.write("New search\n")

    sonar_ranges = np.linspace(3, 6, 8)
    ballasts = [True, False]
    initial_positions = [(1,1), (24,24)]

    
    for ballast in [True]:
        dt = 0.1 # time step
        T = 10 # total time
        n = int(T / dt) # number of time steps
        damper = 2 # multiplier for omega
        mass = 11800
        lat_long = (37.5, 21.5)

        start_position = (0, 0, -2000)
        start_velocity = (-10, 10, 0.1)

        print('Values have been initialized')
        print(f'dt: {dt}, T: {T}, n: {n}, mass: {mass}')
        print(f'Starting position: {start_position}, starting velocity: {start_velocity}')

        # Euler-Maruyama method
        print('Starting Euler-Maruyama method...')

        paths = []
        num_paths = 1000

        for i in range(num_paths):
            if i % 100 == 0:
                print(f'Iteration {i} of {num_paths}')
            # Generate Wiener process
            sd = np.sqrt(dt)

            dW0 = get_wiener_deltas(sd, n)
            dW1 = get_wiener_deltas(sd, n)
            dW2 = get_wiener_deltas(sd, n)

            # W = np.zeros((n - 1, 3))
            W = np.zeros((n, 3))

            W[:, 0] = dW0
            W[:, 1] = dW1
            W[:, 2] = dW2

            positions, velocities, accelarations = euler_maryama(n, dt, mass, W, damper, start_position, start_velocity, lat_long, ballast, log_results = False)

            paths.append(positions)

        for sonar_range in sonar_ranges:
            for initial_position in [(1,1)]:
                print(f"Sonar range: {sonar_range}, Ballast: {ballast}, Initial position: {initial_position}")
                main(paths, sonar_range, ballast, initial_position)
                print("\n\n")