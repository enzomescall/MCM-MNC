import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


phi = np.load("./sim0/phi.npy")
# phi = [timestep, x, y, z]

print(phi.shape)


# Integrate the z axis of phi
phi = np.sum(phi, axis=3)

phi = -phi
fig, ax = plt.subplots()

fig.colorbar(ax.imshow(phi[5, :, :]))

def update(frame):
    ax.clear()
    ax.imshow(phi[frame, :, :])
    ax.set_title(f"Time Step: {frame}")
    
ani = FuncAnimation(fig, update, frames=range(phi.shape[0]), interval=100)

plt.show()