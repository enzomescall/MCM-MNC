import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

# Define the 3D vector field function based on given equations
def vector_field_3d(X, t):
    x, y, z = X
    dxdt = np.cos(y) - z
    dydt =-np.sin(x) - z
    dzdt = np.sin(x) + np.cos(y)
    return [dxdt, dydt, dzdt]

# Generate grid of points in 3D space
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
z = np.linspace(-5, 5, 20)
X, Y, Z = np.meshgrid(x, y, z)

# Calculate the 3D vector field values at each point
U, V, W = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)
for i in range(len(x)):
    for j in range(len(y)):
        for k in range(len(z)):
            x_val, y_val, z_val = X[i, j, k], Y[i, j, k], Z[i, j, k]
            vec_field = vector_field_3d([x_val, y_val, z_val], 0)
            U[i, j, k] = vec_field[0]
            V[i, j, k] = vec_field[1]
            W[i, j, k] = vec_field[2]

# Plot the 3D vector field
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W, length=0.2, normalize=True, color='blue')
ax.set_title('3D Vector Field')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()

# Define initial conditions for 3D trajectory
initial_conditions_3d = [0.5, 0.5, 3.0]

# Define the time parameter for simulation
t = np.linspace(0, 5, 100)

# Integrate the system of ODEs to simulate 3D trajectory
trajectory_3d = np.array(odeint(vector_field_3d, initial_conditions_3d, t))

# Plot the 3D trajectory in the vector field
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W, length=0.2, normalize=True, color='blue', alpha=0.3)
ax.plot(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], 'r-', label='3D Trajectory')
ax.scatter(initial_conditions_3d[0], initial_conditions_3d[1], initial_conditions_3d[2], color='red', marker='o', label='Initial Position')
ax.set_title('3D Trajectory in Vector Field')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
plt.show()
