import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck, boundary, delta_function, potential_from_data
from mpl_toolkits.mplot3d import Axes3D

"""
Solve the Fokker-Planck equation

Arguments:
    temperature     temperature of the surrounding bath (scalar or vector)
    drag            drag coefficient (scalar or vector or function)
    extent          extent (size) of the grid (vector)
    resolution      spatial resolution of the grid (scalar or vector)
    potential       external potential function, U(ndim -> scalar)
    force           external force function, F(ndim -> ndim)
    boundary        type of boundary condition (scalar or vector, default: reflecting)

Translation:
drift/mu -> potential
sigma/diffusion -> drag (or 1/drag unsure)
"""

nm = 1e-9
viscosity = 8e-4
radius = 50*nm
drag = 6*np.pi*viscosity*radius*20

"""
Arguments:
    grid     list of grid arrays along each dimension
    data     potential data
"""

# Generate a grid of points in 2d space
x = np.linspace(-1e-19, 1e-19, 200)
y = np.linspace(-1e-19, 1e-19, 200)

# Create a 2D vector field on these points
gx, gy = np.meshgrid(x, y)

def vector_field_2d(gx, gy):
    return  gx - gy

grid = np.array([x, y])
data = vector_field_2d(gx, gy)



U = potential_from_data(grid, data) 

sim = fokker_planck(temperature=300,
                    drag=drag,
                    extent=[600*nm, 600*nm],
                    resolution=10*nm,
                    boundary=boundary.reflecting,
                    potential=U)

### time-evolved solution
pdf = delta_function((-150*nm, -150*nm))
p0 = pdf(*sim.grid)

Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 2e-3, Nsteps=Nsteps)

### animation
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), constrained_layout=True)

surf = ax.plot_surface(*sim.grid/nm, p0, cmap='viridis')

ax.set_zlim([0,np.max(Pt)/3])
ax.autoscale(False)

def update(i):
    global surf
    surf.remove()
    surf = ax.plot_surface(*sim.grid/nm, Pt[i], cmap='viridis')

    return [surf]

anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
ax.set(xlabel='x (nm)', ylabel='y (nm)', zlabel='normalized PDF')

plt.show()