import gsw
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Sensor():

    def __init__(self, long, lat, temp, depth, speed, angle):


        self.longitude = long
        self.latitude = lat
        self.temperature = temp
        self.depth = depth
        self.current_speed = speed
        self.current_angle = angle

    def interp_temperature(self, depth):
        """
        Finds the temperature near this sensor but at different depths.
        """
        
        if depth <= -1000:
            return 4
        else:
            return ((self.temperature - 4) / (self.depth + 1000)) * (depth + 1000) + 4

    def interp_salinity(self, depth):
        """
        Finds the salinity near this sensor at different depths.
        """

        if depth <= -1000:
            return 38.6
        else:
            return ((38.9 - 38.6) / (1000)) * (depth + 1000) + 38.6

    def distance(self, lat, long):
        """
        Returns the distance between this sensor and the given latitude/longitude coordinates.
        """

        return ((self.latitude - lat)**2 + (self.longitude - long)**2) ** 0.5

    def __str__(self):
        return str("Latitude: " + str(self.latitude) + " Longitude: " + str(self.longitude) + " Temperature (C): " + str(self.temperature) + 
                   " Depth (m): " + str(self.depth) + " Current speed (m/s): " + str(self.current_speed) + " Current angle (deg): " + 
                   str(self.current_angle))
    
root = Path("files/order_65868_unrestricted")
files = root.glob("*.txt")

sensors = []
longs = []

acc = 0

for f in files:
    
    data = pd.read_table(f)

    longitude = data["Longitude [degrees_east]"][0]
    latitude = data["Latitude [degrees_north]"][0]

    if longitude in longs:
        continue

    if "Temp [degC]" not in data:
        continue
    idx = data["Temp [degC]"].first_valid_index()
    if idx is None:
        continue
    temp = data["Temp [degC]"][idx]

    idx = data["UnspCurrSpd [m/s]"].first_valid_index()
    if idx is None:
        continue
    speed = data["UnspCurrSpd [m/s]"][idx]

    idx = data["UnspCurrDir [deg]"].first_valid_index()
    if idx is None:
        continue
    angle = data["UnspCurrDir [deg]"][idx]

    idx = data["DepBelowSurface [m]"].first_valid_index()
    depth = data["DepBelowSurface [m]"][idx]

    sensors.append(Sensor(longitude, latitude, temp, -depth, speed, angle))
    longs.append(longitude)

def buoyancy(rho, ballast = True):
    "Buoyancy equation"
    return 9.81 * (rho * 11.8 - (11800 + 500 * ballast))

def drag_force(rho, relative_velocity, area=15, coeff=0.1):
    "Drag force equation"
    return 0.5 * rho * relative_velocity**2 * area * coeff

def calculate_forces(latitude, depth, velocity, conditions, ballast=True):
    """
    Calculates the forces acting on the submersible.

    ARGS:
        Latitude: the latitude coordinate of the vessel
        Depth: the depth in meters (should be negative, ex: -500m)
        Velocity: the craft's velocity in m/s
            Should be a 3-vector with velocity in the x/y/z directions.
            Positive velocity reflects eastward, northward, and upward motion respectively.
        Conditions: the nearest sensor to the craft.
        Ballast: whether or not the ballast is still attached to the craft

    RETURNS:
        forces: 3-vector with forces in x/y/z directions, in Newtons.
        Recall that the mass of the craft is 11800 kg to convert to acceleration.
    """

    x_force = 0
    y_force = 0
    z_force = 0
    
    pressure = gsw.conversions.p_from_z(depth, latitude)
    rho = gsw.density.rho(conditions.interp_salinity(depth), conditions.interp_temperature(depth), pressure)
    z_force += buoyancy(rho, ballast)

    angle = np.radians(conditions.current_angle)
    speed = conditions.current_speed
    
    x_speed = speed * np.cos(angle) - velocity[0]
    y_speed = speed * np.sin(angle) - velocity[1]
    z_speed = -velocity[2]

    x_force += np.sign(x_speed) * drag_force(rho, x_speed)
    y_force += np.sign(y_speed) * drag_force(rho, y_speed)
    z_force += np.sign(z_speed) * drag_force(rho, z_speed)

    return np.array([x_force, y_force, z_force])

def position_to_latlong(position: (float, float), start_lat_long: (float, float)):
    """
    Converts a position to latitude and longitude.
    Assuming starting latitude and longitude are 
    in the middle of the Ionian sea
    Assuming that the algorithm starts at (0, 0)
    """
    lat, long = start_lat_long

    meters_lat = 111132.92 - 559.82 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat) - 0.0023 * np.cos(6 * lat)
    meters_long = 111412.84 * np.cos(long) - 93.5 * np.cos(3 * long) + 0.118 * np.cos(5 * long)

    current_lat = lat + (position[1] / meters_lat)
    current_long = long + (position[0] / meters_long)

    return current_long, current_lat
       

def force(position: (float, float, float), velocity: (float, float, float), start_lat_long: (float, float), ballast = True, log_results = False):
    
    depth = position[2]
    
    long, lat = position_to_latlong(position[:2], start_lat_long)

    dist = np.inf
    conditions = sensors[0]
    for s in sensors:
        displacement = s.distance(lat, long)
        if displacement < dist:
            dist = displacement
            conditions = s
        elif dist == np.inf:
            print("failed", long, lat, depth, velocity)
            raise ValueError("No sensors in range")

    if log_results:
        print(f'Calculating forces with lat: {lat}, long: {long}, depth: {depth}')
        print(f'And conditions: {conditions}')
    return calculate_forces(lat, depth, velocity, conditions, ballast)


# MAIN
if __name__ == "__main__":
    # EXAMPLE USE CASE

    # I start by defining the location of the craft, and its initial velocity.
    lat = 37.22
    long = 15.3
    depth = -700
    velocity = [0.25, 0.25, -0.1]

    # Plotting sensor locations
    fig, ax = plt.subplots()
    for sensor in sensors:
        ax.scatter(sensor.longitude, sensor.latitude)

    sensor = sensors[0]
    lat, longn = sensors[1].latitude, sensors[1].longitude

    # Print lat and long of first two sensors
    print(lat, long)
    print(sensor.latitude, sensor.longitude)


    print(sensor.distance(lat, long))
    print(sensor.distance(38.22, 16.3))
    print(sensor.distance(position_to_latlong((0, 0), (38.22, 16.3))))

    ax.scatter(long, lat, color="red")
    plt.show()
