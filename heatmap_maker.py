import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck, boundary, gaussian_pdf

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
drift/mu -> force
sigma/diffusion -> drag (or 1/drag)
"""

initial_values = (0, 0, 0)

def mu(x, y, z):
    return (x, y, z)    

def sigma(x, y, z):
    return (x, y, z)

sim = fokker_planck(temperature=10,
                    drag=sigma,
                    extent=200,
                    resolution=1,
                    force=mu,
                    boundary=boundary.absorbing)

### steady-state solution
steady = sim.steady_state()

### time-evolved solution
w = 30
pdf = gaussian_pdf(0, w)
p0 = pdf(sim.grid[0])

Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 1e-3, Nsteps=Nsteps)

### animation
fig, ax = plt.subplots()

ax.plot(sim.grid[0], steady, color='k', ls='--', alpha=.5)
ax.plot(sim.grid[0], p0, color='red', ls='--', alpha=.3)
line, = ax.plot(sim.grid[0], p0, lw=2, color='C3')

def update(i):
    line.set_ydata(Pt[i])
    return [line]

anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
ax.set(xlabel='x', ylabel='normalized PDF')
ax.margins(x=0)

plt.show()