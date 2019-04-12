import numpy as np
from scipy.integrate import solve_ivp
import scipy.ndimage

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

import progressbar

### initial conditions
t0 = 0
tf = 10

### define grid
n_x = 100
n_y = 100

### physical constants/parameters
T = 6 # tension, N
l = 1 / n_x # distance between grid points, m
m = 0.1 # mass of single grid point, kg
b = 0.02 # resistance coefficient

### grid coordinates
x = np.linspace(-1, 1, n_x)
y = np.linspace(-1, 1, n_y)

### z values
z_grid = np.empty((n_y, n_x))

### initial perturbation
for i in range(n_y):
    y_val = y[i]
    for j in range(n_x):
        x_val = x[j]
        ### 2D gaussian function
        z_grid[i][j] = 0.3 * np.exp(-1 * (x_val - 0.1)**2 / 0.15)
        z_grid[i][j] *= np.exp(-1 * (y_val - 0.1)**2 / 0.15)
        z_grid[i][j] += 0.3 * np.exp(-1 * (x_val + 0.4)**2 / 0.15)
        z_grid[i][j] *= np.exp(-1 * (y_val + 0.2)**2 / 0.15)

zmax = 1.1 * np.abs(np.max(z_grid))

init_state = np.zeros(2 * n_x * n_y)
init_state[:n_x * n_y] = z_grid.flatten()

X, Y = np.meshgrid(x, y)

def derivs(t, state):
    z = np.reshape(state[:n_x * n_y], (n_y, n_x)) # reshape z array to represent true grid
    dzdt = state[n_x * n_y:] # first derivatives
    
    ### left, right, top, and bottom z coordinates
    z_l = np.zeros((n_y, n_x))
    z_r = np.zeros((n_y, n_x))
    z_t = np.zeros((n_y, n_x))
    z_b = np.zeros((n_y, n_x))

    #r_squared = X**2 + Y**2
    #z[r_squared > 0.9] = 0
    ### take z coordinate of point to left of current point
    ### for leftmost point, this is zero
    z_l[:,range(1, n_x)] = z[:,range(0, n_x - 1)]
    
    ### repeat this for the other three directions
    z_r[:,range(0, n_x - 1)] = z[:,range(1, n_x)]
    z_t[1:n_y] = z[0:n_y - 1]
    z_b[0:n_y- 1] = z[1:n_y]

    ### delta z's for all four directions
    dz_r = z - z_r
    dz_l = z - z_l
    dz_t = z - z_t
    dz_b = z - z_b

    ### forces
    F_r = -1 * T * (dz_r / l)
    F_l = -1 * T * (dz_l / l)
    F_t = -1 * T * (dz_t / l)
    F_b = -1 * T * (dz_b / l)

    ### net force
    F = F_r + F_l + F_t + F_b

    ### acceleration
    a = (F / m).flatten()

    a -= b * dzdt / m

    ### create return array and fill it
    ret = np.empty(2 * n_x * n_y)
    ret[:n_x * n_y] = dzdt
    ret[n_x * n_y:] = a

    return ret

### boolean arrays for masking
truth_x = np.full(n_x, False, dtype=bool)
truth_y = np.full(n_y, False, dtype=bool)
for i in range(0, n_x, 4):
    truth_x[i] = True
for i in range(0, n_y, 4):
    truth_y[i] = True

### solve diff. eq.
sol = solve_ivp(derivs, (t0, tf), init_state, dense_output = True)
print(sol.message)

fig = plt.figure()
ax = fig.gca(projection='3d')

z_i = sol.sol(t0)
z_i = np.reshape(z_i[:n_x * n_y], (n_y, n_x))

X = X[truth_y][:,truth_x]
Y = Y[truth_y][:,truth_x]

plot = ax.plot_surface(X, Y, z_i[truth_y][:,truth_x], cmap=cm.gnuplot)#cmap=cm.viridis)

def update(framenumber, nframes, plot, bar):
    bar.update(framenumber)
    t = (framenumber / nframes) * (tf - t0)
    res = sol.sol(t)
    z = np.reshape(res[:n_x * n_y], (n_y, n_x))
    ax.clear()
    plot = ax.plot_surface(X, Y, z[truth_y][:,truth_x], 
            cmap='viridis', vmin=-1*zmax, vmax=zmax)
    ax.set_zlim3d(-0.15, 0.15)
    return plot,

interval = 20 # ms
nframes = int(tf / (interval * 1e-3))

bar = progressbar.ProgressBar(maxval=nframes, widgets=[progressbar.Bar('=', 
                                    '[', ']'), ' ', progressbar.Percentage()])
bar.start()

ani = animation.FuncAnimation(fig, update, nframes, fargs=(nframes, plot, bar),
                              interval=interval, blit=False)

ani.save("animation.mp4")
print("\nFinished")
