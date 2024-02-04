import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


x_width = 10
y_width = 10
z_width = 10

num_points = 100

grid_x = np.linspace(0, x_width, num_points)
grid_y = np.linspace(0, y_width, num_points)
grid_z = np.linspace(0, z_width, num_points)

# grid = np.array((grid_x, grid_y, grid_z))


sigma = 1.  # standard deviation
mu = 10.  # mean
tau = .05  # time constant

dt = .001  # time step
T = 1.  # total time
n = int(T / dt)  # number of time steps
t = np.linspace(0., T, n)  # vector of times

CENTER = [1, 1, 1]
SD = 1

drift = np.zeros((num_points, num_points, num_points))
diffusion = np.zeros((num_points, num_points, num_points))

def calc_drift(drift, position, velocity):
    #vt + 0.5at^2
    return position + velocity * dt + 0.5 * drift[position] * (dt**2)


def calc_diffusion(diffusion, position, velocity):
    pass

def get_wiener_deltas(sd, interval_count):
    wieners = np.random.normal(0, sd, interval_count)
    wiener_deltas = np.zeros(wieners.size - 1)
    for i in range(wieners.size - 1):
        wiener_deltas[i] = wieners[i + 1] - wieners[i]
    return wiener_deltas

dW0 = get_wiener_deltas(dt, n)
dW1 = get_wiener_deltas(dt, n)
dW2 = get_wiener_deltas(dt, n)

W = np.zeros((n - 1, 3))

W[:, 0] = dW0
W[:, 1] = dW1
W[:, 2] = dW2


start_position = np.array([0, 0, 0])
velocity = np.array([0.1, 0.1, 0.1])

# print(W[4], W[3])

positions = np.zeros((n, 3))

positions[0] = start_position

for i in positions.shape[0]:
    positions[i + 1] = calc_drift(positions[i]) * dt + calc_diffusion(positions[i]) * W[i]
    
