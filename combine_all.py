from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from buoyancy import position_to_latlong
import sys
sys.path.append('ADE-python/structured_grid')

from path_tester import complete_search, stepwise_search, lg, lt

"""
Combining all parts of the process to make the master heatmaps\

Plot 1: heatmap of potential submersible positions v.s. simulated vessel path

Plot 2: heatmaps of pdf moving around and search path adapting

Plot 3: heatmap plus search path plus simulated vessel path
"""

def plot_1(lat_longs):
    # Load data
    positions = np.load('timestep.npy')

    # convert rangee to latlong
    top_right = position_to_latlong((210,210), lat_longs)
    bottom_left = position_to_latlong((-10,-10), lat_longs)

    range = [[top_right[0], bottom_left[0]], [bottom_left[1], top_right[1]]]

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.suptitle('Potential v.s. Actual Submersible Position', fontsize=15, y=0.97)

    # Increase spacing between subplots
    fig.subplots_adjust(wspace=0.1)

    plot_path = []
    # Extracting a single path
    for i, step in enumerate(positions):
        if i % 10 == 0:
            x_value, y_value = position_to_latlong((step[0,0], step[1,0]), lat_longs)
            # print(f'Appending step {i} with coords {x_value, y_value}')
            plot_path.append((x_value, y_value))

    plot_path = np.array(plot_path)

    for i, t in enumerate([9, 999]):
        all_x = positions[t, 0]
        all_y = positions[t, 1]

        # convert to latlong
        all_x, all_y = position_to_latlong((all_x, all_y), lat_longs)

        heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=25, density=True, range=range)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        heatmap = gaussian_filter(heatmap, sigma=1)

        axs[i].plot(plot_path[:t, 0], plot_path[:t, 1], color='red', label='Simulated submersible path')

        axs[i].locator_params(axis='x', nbins=3)

        if i == 0:
            axs[i].set_ylabel('Latitude')
            axs[i].legend()
        else:
            # remove y tickers and labels
            axs[i].set_yticklabels([])
            axs[i].set_yticks([])

        axs[i].set_title(f'Elapsed time: {(t+1)//10}s')
        axs[i].imshow(heatmap.T, extent=extent, origin='lower')
        axs[i].set_xlabel('Longitude')

    plt.savefig('potential_actual.png')

def plot_2(lat_longs):
    # Complete search needs (k_generations, n_paths, start_point,
    #                        desired_path_length, heatmap,
    #                        sonar_range, sonar_falloff)
    
    k_generations = 5
    n_paths = 400
    start_point = (47, 49)
    desired_path_length = 6
    sonar_range = 2
    sonar_falloff = 2

    # We will turn timestep.npy into a heatmap
    positions = np.load('timestep.npy')

    # convert rangee to latlong
    top_right = position_to_latlong((210,210), lat_longs)
    bottom_left = position_to_latlong((-10,-10), lat_longs)

    plot_range = [[top_right[0], bottom_left[0]], [bottom_left[1], top_right[1]]]

    heatmaps = [] # heatmaps for each time step
    for t in range(999):
        all_x = positions[t, 0]
        all_y = positions[t, 1]

        # convert to latlong
        all_x, all_y = position_to_latlong((all_x, all_y), lat_longs)

        heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=50, density=True, range=plot_range)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        heatmap = gaussian_filter(heatmap, sigma=1)
        heatmaps.append(heatmap)

    # Run search algorithm
    # top_path_1 = stepwise_search(k_generations, n_paths, start_point, desired_path_length, heatmaps[:100], sonar_range, sonar_falloff)[0]
    # top_path_2 = stepwise_search(k_generations, n_paths, start_point, desired_path_length, heatmaps[:500], sonar_range, sonar_falloff)[0]
    # top_path_3 = stepwise_search(k_generations, n_paths, start_point, desired_path_length, heatmaps, sonar_range, sonar_falloff)[0]

    # # save these three paths
    # np.save('top_path_1.npy', top_path_1)
    # np.save('top_path_2.npy', top_path_2)
    # np.save('top_path_3.npy', top_path_3)

    # load these three paths
    top_path_1 = np.load('top_path_1.npy')
    top_path_2 = np.load('top_path_2.npy')
    top_path_3 = np.load('top_path_3.npy')

    # Plot the results
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    plt.figure(figsize=(12, 3.9))
    plt.tight_layout()
    plt.suptitle('Example of PDF Moving and Search Path Adapting', fontsize=15, y = 0.97)

    # Subplot 1
    plt.subplot(1, 3, 1)
    plt.imshow(heatmaps[99].T, interpolation='nearest', origin="lower", extent=(lg(0), lg(50), lt(0), lt(50)))
    plt.scatter(lg(start_point[0]), lt(start_point[1]), color='white', marker='.', label='Initial position', s=250, zorder=3)
    plt.scatter(lg(top_path_1[-1][0]), lt(top_path_1[-1][1]), color='black', marker='X', label=f'Final position at: {len(top_path_1)} steps', s=150, zorder=2)
    plt.plot([lg(x) for x, y in top_path_1], [lt(y) for x, y in top_path_1], color='red', label='Search path', linewidth=3, zorder=1)
    plt.title('Initial Search Path')
    plt.xlabel('Long.')
    plt.ylabel('Lat.')
    plt.legend(loc='upper left')

    # Subplot 2
    plt.subplot(1, 3, 2)
    plt.imshow(heatmaps[500].T, interpolation='nearest', origin="lower", extent=(lg(0), lg(50), lt(0), lt(50)))
    plt.scatter(lg(start_point[0]), lt(start_point[1]), color='white', marker='.', s=250, zorder=3)
    plt.scatter(lg(top_path_2[-1][0]), lt(top_path_2[-1][1]), color='black', marker='X', label=f'Final position at: {len(top_path_2)} steps', s=150, zorder=2)
    plt.plot([lg(x) for x, y in top_path_2], [lt(y) for x, y in top_path_2], color='red', linewidth=3, zorder=1)
    plt.title('Search Path at 500th timestep')
    plt.xlabel('Long.')
    plt.legend(loc='upper left')

    # Subplot 3
    plt.subplot(1, 3, 3)
    plt.imshow(heatmap.T, interpolation='nearest', origin="lower", extent=(lg(0), lg(50), lt(0), lt(50)))
    plt.scatter(lg(start_point[0]), lt(start_point[1]), color='white', marker='.', s=250, zorder=3)
    plt.scatter(lg(top_path_3[-1][0]), lt(top_path_3[-1][1]), color='black', marker='X', label=f'Final position at: {len(top_path_3)} steps', s=150, zorder=2)
    plt.plot([lg(x) for x, y in top_path_3], [lt(y) for x, y in top_path_3], color='red', linewidth=3, zorder=1)
    plt.xlabel('Long.')
    plt.title('Final Search Path')
    plt.legend(loc='upper left')

    plt.savefig('search_path_adapting.png')

    

def plot_3(lat_longs):
    # Simulating a sub, heatmap, and search and then seeing if the vessel finds the sub

    start_point = (47, 49)

    # load the final path
    top_path_3 = np.load('top_path_3.npy')
    # Load data
    positions = np.load('timestep.npy')
    path = np.load('timestep2.npy')

    # convert rangee to latlong
    top_right = position_to_latlong((210,210), lat_longs)
    bottom_left = position_to_latlong((-10,-10), lat_longs)

    plot_range = [[top_right[0], bottom_left[0]], [bottom_left[1], top_right[1]]]

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.suptitle('Potential v.s. Actual Submersible Position', fontsize=15, y=0.97)

    # Increase spacing between subplots
    fig.subplots_adjust(wspace=0.1)

    plot_path = []
    # Extracting a single path
    for i, step in enumerate(positions):
        if i % 10 == 0:
            x_value, y_value = position_to_latlong((step[0,0], step[1,0]), lat_longs)
            # print(f'Appending step {i} with coords {x_value, y_value}')
            plot_path.append((x_value, y_value))

    plot_path = np.array(plot_path)

    for t in [999]:
        all_x = positions[t, 0]
        all_y = positions[t, 1]

        # convert to latlong
        all_x, all_y = position_to_latlong((all_x, all_y), lat_longs)

        heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=50, density=True, range=plot_range)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        heatmap = gaussian_filter(heatmap, sigma=1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    plt.figure(figsize=(12, 3.9))
    plt.tight_layout()
    plt.suptitle('Search Vessel Looking for Submersible', fontsize=15, y = 0.97)

    # Subplot 1: path of sub
    plt.subplot(1, 2, 1)
    plt.imshow(heatmap.T, interpolation='nearest', origin="lower", extent=(lg(0), lg(50), lt(0), lt(50)))       
    print(plot_path[:, 0], plot_path[:, 1])
    plt.plot(plot_path[:, 0], plot_path[:, 1], color='blue', label='Submersible path', linewidth=3, zorder=1)

    # Subplot 2
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap.T, interpolation='nearest', origin="lower", extent=(lg(0), lg(50), lt(0), lt(50)))
    plt.scatter(lg(start_point[0]), lt(start_point[1]), color='white', marker='.', s=250, zorder=3)
    plt.scatter(lg(top_path_3[-1][0]), lt(top_path_3[-1][1]), color='black', marker='X', label=f'Final position at: {len(top_path_3)} steps', s=150, zorder=2)
    plt.plot([lg(x) for x, y in top_path_3], [lt(y) for x, y in top_path_3], color='red', linewidth=3, zorder=1)
    plt.xlabel('Long.')
    plt.title('Final Search Path')
    plt.legend(loc='upper left')

    plt.savefig('finding_sub.png')

if __name__ == "__main__":
    lat_longs = (38.22, 16.3) 

    plot_3(lat_longs)
