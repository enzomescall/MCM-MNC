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
    return 1
    # return 0.5 * rho * relative_velocity**2 * area * coeff


# def calculate_forces(latitude, depth, velocity, conditions, ballast=True):
def calculate_forces(latitude, depth, conditions, ballast=True):
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

    # x_speed = speed * np.cos(angle) - velocity[0]
    # y_speed = speed * np.sin(angle) - velocity[1]
    # z_speed = -velocity[2]
    
    x_speed = (speed * np.cos(angle))/100 # dampen everything into centinewtons
    y_speed = (speed * np.sin(angle))/100
    z_speed = (z_force)/100

    # x_force += np.sign(x_speed) * drag_force(rho, x_speed)
    # y_force += np.sign(y_speed) * drag_force(rho, y_speed)
    # z_force += np.sign(z_speed) * drag_force(rho, z_speed)

    # return np.array([x_force, y_force, z_force]), rho

    return np.array([x_speed, y_speed, z_speed]), rho

# EXAMPLE USE CASE

# I start by defining the location of the craft, and its initial velocity.
# lat = 37.22
# long = 15.3
# depth = -700
# velocity = [0.25, 0.25, -0.1]

# The craft has latitude 37.22 and longitude 15.3, and is 700 meters below the surface.
# Its total velocity is 0.37 m/s; 0.25 eastward, 0.25 northward, 0.1 downward.

# This finds the nearest sensor. Very naive method but whatever

def forces(lat, long, depth, ballast=True):
    dist = float("inf")
    for s in sensors:
        displacement = s.distance(lat, long)
        if displacement < dist:
            dist = displacement
            conditions = s

    # And this gets the forces! Yay!
    # Feel free to divide this by 11800 to get the acceleration.
    # Set ballast=True if ballast is on-board and False if it's not.
    # Note that the craft is slightly decelerating in XY directions.
    # If it has ballast, it's sinking slowly. If not, it's rising pretty fast.
    force = calculate_forces(lat, depth, conditions, ballast)
    
    return force
