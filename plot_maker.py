import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from buoyancy import position_to_latlong

# Load data
positions = np.load('timestep.npy')

# convert rangee to latlong
top_right = position_to_latlong((210,210), (38.22, 16.3))
bottom_left = position_to_latlong((-10,-10), (38.22, 16.3))

range = [[top_right[0], bottom_left[0]], [bottom_left[1], top_right[1]]]

print(range)

fig, axs = plt.subplots(1, 3, figsize=(8, 2.8))
fig.suptitle('Heatmap of Potential Submersible Positions', fontsize=15, y=0.95)

# Increase spacing between subplots
fig.subplots_adjust(wspace=0.3)

for i, t in enumerate([9, 499, 999]):
    all_x = positions[t, 0]
    all_y = positions[t, 1]

    # convert to latlong
    all_x, all_y = position_to_latlong((all_x, all_y), (38.22, 16.3))

    heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=25, density=True, range=range)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    heatmap = gaussian_filter(heatmap, sigma=1)

    if i == 0:
        axs[i].set_ylabel('Latitude')
    else:
        # remove y tickers and labels
        axs[i].set_yticklabels([])
        axs[i].set_yticks([])

    axs[i].set_title(f'Elapsed time: {(t+1)//10}s')
    axs[i].imshow(heatmap, extent=extent, origin='lower')
    axs[i].set_xlabel('Longitude')

plt.savefig('three_heatmaps.png')

