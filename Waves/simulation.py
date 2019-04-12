import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation

### number of masses
n = 250

### initialize x positions of masses
x_l = 1.0 / (n + 2)
x_r = 1.0 - x_l

x = np.linspace(x_l, x_r, n)

### constants
L = 1.0 / n # distance between masses
T = 1.0 # tension
omega = 20 # angular frequency
m = (4 * T) / (omega**2 * L) # mass of each mass, set using desired frequency and constants (and math)
m *= (1 * np.pi) / (2 * (n + 1))

t0 = 0
tf = 20

y_i = np.zeros(n)
acc_i = np.zeros(n)

def derivs(t, state):
    y = state[:n]
    dydt = state[n:]
    y_rr = 0.001 * np.sin(omega * t / 2) * np.cos(omega * t / 3)
    y_l = np.append([0], y[:-1])
    y_r = np.append(y[1:], [y_rr])
    sin_theta_p = y - y_l
    denominator = np.sqrt((y - y_l)**2 + L**2)
    sin_theta_p /= denominator
    sin_theta_r = y_r - y
    denominator = np.sqrt((y_r - y)**2 + L**2)
    sin_theta_r /= denominator
    F = (-1 * T * sin_theta_p) + (T * sin_theta_r)
    acc = F / m
    ret = np.empty(n * 2)
    ret[:n] = dydt
    ret[n:] = acc
    return ret

state = np.zeros(n * 2)
state[:n] = y_i
state[n:] = acc_i
sol = solve_ivp(derivs, (t0, tf), state, dense_output=True)

print(sol.message)

n_rows, _ = (sol.y).shape
fig = plt.figure()
ax = plt.axes(xlim=(-0.2, 1.2), ylim=(-0.02, 0.02))
line, = ax.plot([], [], 'o', color='black', markersize=0.5)

nframes = 60 * (tf - t0)
t = np.linspace(t0, tf, nframes)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    row = sol.sol(t[i])
    y = row[:n]
    line.set_data(x, y)
    return line,

interval = 20 # ms
nframes = int(tf / (interval * 1e-3))

ani = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nframes, interval=interval, blit=True)

ani.save("animation.mp4")
