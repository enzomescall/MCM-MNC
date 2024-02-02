import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the vector field function based on given equations
def vector_field(X, t):
    x, y = X
    dxdt = np.cos(y)
    dydt = -np.sin(x)
    return [dxdt, dydt]

# Generate grid of points in the xy-plane
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)

# Calculate the vector field values at each point
U, V = np.zeros_like(X), np.zeros_like(Y)
for i in range(len(x)):
    for j in range(len(y)):
        x_val = X[i, j]
        y_val = Y[i, j]
        vec_field = vector_field([x_val, y_val], 0)
        U[i, j] = vec_field[0]
        V[i, j] = vec_field[1]

# Plot the vector field
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, scale=20, color='blue')
plt.title('Vector Field')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Define initial conditions for particle
initial_conditions = [-1, 0.5]

# Define the time parameter for simulation
t = np.linspace(0, 5, 100)

# Integrate the system of ODEs to simulate particle movement
particle_path = odeint(vector_field, initial_conditions, t)

# Plot the particle's trajectory in the vector field
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, scale=20, color='blue')
plt.plot(particle_path[:, 0], particle_path[:, 1], 'r-', label='Particle Trajectory')
plt.scatter(initial_conditions[0], initial_conditions[1], color='red', marker='o', label='Initial Position')
plt.title('Particle Movement in Vector Field')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
