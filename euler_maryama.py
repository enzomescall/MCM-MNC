from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from buoyancy import force, position_to_latlong

"""
OLD ERIK FUNCTION
"""
@njit
def get_wiener_deltas(sd: float, interval_count):
    wieners = np.random.normal(0, sd, interval_count)
    return wieners
    # wiener_deltas = np.zeros(wieners.size - 1)
    # for i in range(wieners.size - 1):
    #     wiener_deltas[i] = wieners[i + 1] - wieners[i]
    # return wiener_deltas
   
"""
NEW ENZO FUNCTIONS
"""
@njit
def calc_accelaration(force: (float, float, float), mass: float) -> (float, float, float):
    return force / mass

@njit
def stochastic_acceleration(acceleration: (float, float, float), omega: (float, float, float), W: (float, float, float), dt: float) -> (float, float, float):
    return acceleration + omega * W * np.sqrt(dt)

@njit
def omega(acceleration: (float, float, float), damper: float) -> float:
    return damper

@njit
def step_velocity(velocity: (float, float, float),
                  acceleration: (float, float, float),
                    omega: float, W: (float, float, float),
                  dt: float) -> (float, float, float):
    return velocity + acceleration * dt + omega * W * np.sqrt(dt)

@njit
def step_position(position: (float, float, float), velocity: (float, float, float), acceleration: (float, float, float), dt: float) -> (float, float, float):
    return position + velocity * dt + 0.5 * acceleration * dt**2

def euler_maryama(n: int,
                  dt: float,
                  mass: float,
                  W: np.ndarray,
                  damper: float,
                  start_position: (float, float, float),
                  start_velocity: (float, float, float),
                  start_lat_long: (float, float),
                  ballast: bool = True,
                  log_results = False) -> (np.ndarray, np.ndarray):
    # Initialize
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))

    velocities[0] = start_velocity
    positions[0] = start_position
    accelerations = []

    omega_value = damper

    for t in range(n-1):
        if log_results:
            print(f'--------------------Iteration {t}/{n-1}--------------------')
        # Calculate forces: force(lat, long, depth, velocity, ballast = True)
        forces = force(positions[t], velocities[t], start_lat_long, ballast, log_results = log_results) 

        # need acceleration to calculate omega
        acceleration = calc_accelaration(forces, mass)
        
        # need omega to calculate stochastic acceleration
        # omega_value = omega(acceleration, damper)

        # need stochastic acceleration to calculate velocity
        accelerations.append(acceleration)

        # need velocity to calculate position
        velocities[t + 1] = step_velocity(velocities[t], acceleration, omega_value,W[t],  dt)

        # step position
        position_curr_v = step_position(positions[t], velocities[t], acceleration, dt)
        position_next_v = step_position(positions[t], velocities[t + 1], acceleration, dt)

        # average between current and next velocity
        positions[t + 1] = (position_curr_v + position_next_v)/2

        if log_results:
            print(f'Forces: {forces}')
            print(f'Acceleration: {acceleration}')
            print(f'Omega: {omega_value}, with damper: {damper}')
            print(f'Wiener impact: {np.sqrt(dt) * omega_value * W[t]}')
            print(f'Velocity: {velocities[t + 1]}')
            print(f'Position: {positions[t + 1]}')

    return positions, velocities, accelerations

if __name__ == "__main__":
    # Initialize parametersy

    dt = 0.1 # time step
    T = 10000 # total time
    n = int(T / dt) # number of time steps
    damper = 0.02 # multiplier for omega
    mass = 11800
    lat_long = (37.5, 21.5)

    start_position = (0, 0, -2000)
    start_velocity = (-1, 1, 0.1)

    print('Values have been initialized')
    print(f'dt: {dt}, T: {T}, n: {n}, mass: {mass}')
    print(f'Starting position: {start_position}, starting velocity: {start_velocity}')

    # Euler-Maruyama method
    print('Starting Euler-Maruyama method...')

    paths = []
    num_paths = 1

    for i in range(num_paths):
        if i % 10 == 0:
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

        positions, velocities, accelarations = euler_maryama(n, dt, mass, W, damper, start_position, start_velocity, lat_long, log_results = False)

        paths.append(positions)
        # # Ask user if they want to plot the results
        # user_input = input('Do you want to plot the results? (y/n): ')
        # if user_input.lower() != 'y':
        #     print('Exiting...')
        #     exit()

    # Plotting the results
    print('Plotting the results...')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, path in enumerate(paths):
        # convert all x and y values to lat and long
        lat_longs = np.array([position_to_latlong((x, y), lat_long) for x, y in path[:, :2]])

        # change the color of the path based on index
        ax.plot(lat_longs[:, 0], lat_longs[:, 1], path[:, 2], label=f'Path {i}', color=(i/num_paths, 0, 1 - i/num_paths))

    # Plotting the initial position after conversion to latlong
    start_lat_long = position_to_latlong(start_position[:2], lat_long)
    ax.scatter(start_lat_long[0], start_lat_long[1], start_position[2], color='red', marker='o', label='Initial position')

    # remove y tickers and labels
    ax.set_ylabel('Latitude')

    ax.set_xlabel('Longitude')
    ax.set_zlabel('Depth (m)')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax.set_title('Example of a potential path underwater submersible', fontsize=15)

    plt.savefig('3D_path.png')
    plt.show()

    # Making a heatmap of the paths
    print('Appending arrays...')

    timestep = []

    for i in range(n):
        if i % 100 == 0:
            print(f'Appending timestep {i} of {n}')

        all_x = []
        all_y = []

        for path in paths:
            all_x = np.append(all_x, path[i, 0])
            all_y = np.append(all_y, path[i, 1])

        timestep.append((all_x, all_y))

    # save timestep to a file
    np.save('timestep2.npy', timestep)
