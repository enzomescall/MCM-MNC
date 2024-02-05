from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from buoyancy import force

"""
OLD ERIK FUNCTION
"""
def get_wiener_deltas(sd: float, interval_count):
    wieners = np.random.normal(0, sd, interval_count)
    wiener_deltas = np.zeros(wieners.size - 1)
    for i in range(wieners.size - 1):
        wiener_deltas[i] = wieners[i + 1] - wieners[i]
    return wiener_deltas
   
"""
NEW ENZO FUNCTIONS
"""
def calc_accelaration(force: (float, float, float), mass: float) -> (float, float, float):
    return force / mass

def omega(acceleration: (float, float, float), damper: float) -> float:
    return np.linalg.norm(acceleration) * damper

def step_velocity(t: int,
                  velocity: (float, float, float),
                  acceleration: (float, float, float),
                  omega: (float),
                  W: np.ndarray,
                  dt: float) -> (float, float, float):
    return velocity + acceleration * dt + np.sqrt(dt) * omega * W[t]

def step_position(position: (float, float, float), velocity: (float, float, float), dt: float) -> (float, float, float):
    return position + velocity * dt

def euler_maryama(n: int,
                  dt: float,
                  mass: float,
                  W: np.ndarray,
                  damper: float,
                  start_position: (float, float, float),
                  start_velocity: (float, float, float),
                  start_lat_long: (float, float) = (38.22, 16.3),
                  log_results = False) -> (np.ndarray, np.ndarray):
    # Initialize
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))

    velocities[0] = start_velocity
    positions[0] = start_position

    for t in range(n-1):
        if log_results:
            print(f'--------------------Iteration {t}/{n-1}--------------------')
        # Calculate forces: force(lat, long, depth, velocity, ballast = True)
        forces = force(positions[t], velocities[t], start_lat_long, ballast = True, log_results = log_results) 

        # Calculate acceleration
        acceleration = calc_accelaration(forces, mass)
        
        
        # Calculate omega
        omega_value = omega(acceleration, damper)

        # Step velocity
        velocities[t + 1] = step_velocity(t, velocities[t], acceleration, omega_value, W, dt)

        # Step position
        positions[t + 1] = step_position(positions[t], velocities[t], dt)

        if log_results:
            print(f'Forces: {forces}')
            print(f'Acceleration: {acceleration}')
            print(f'Omega: {omega_value}')
            print(f'Wiener impact: {np.sqrt(dt) * omega_value *W[t]}')
            print(f'Velocity: {velocities[t + 1]}')
            print(f'Position: {positions[t + 1]}')

    return positions, velocities

if __name__ == "__main__":
    # Initialize parametersy

    dt = 1 # time step
    T = 10000.0 # total time
    n = int(T / dt) # number of time steps
    damper = 3 # multiplier for omega
    mass = 11800

    start_position = np.array([0, 0, -2000])
    start_velocity = np.array([0.1, 0.1, 0.1])

    num_points = 100

    print('Values have been initialized')
    print(f'dt: {dt}, T: {T}, n: {n}, mass: {mass}')
    print(f'Starting position: {start_position}, starting velocity: {start_velocity}')

    # Generate Wiener process
    print('Generating Wiener process...')

    sd = np.sqrt(dt)

    dW0 = get_wiener_deltas(sd, n)
    dW1 = get_wiener_deltas(sd, n)
    dW2 = get_wiener_deltas(sd, n)

    W = np.zeros((n - 1, 3))

    W[:, 0] = dW0
    W[:, 1] = dW1
    W[:, 2] = dW2

    print('Wiener process generated')

    # Euler-Maruyama method
    print('Starting Euler-Maruyama method...')

    for i in np.linspace(5, 9, 41):
        print(f'Running Euler-Maruyama method with {i} damper')
        positions, velocities = euler_maryama(n, dt, mass, W, i, start_position, start_velocity)

        print(f'Euler-Maruyama method finished after {n} iterations')

        # # Ask user if they want to plot the results
        # user_input = input('Do you want to plot the results? (y/n): ')
        # if user_input.lower() != 'y':
        #     print('Exiting...')
        #     exit()

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Euler-Maruyama method')
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='red', label='Start')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # save figure in all_the_plots subfolder
        plt.savefig(f'./all_the_plots/plot_omega_{i}.png')
