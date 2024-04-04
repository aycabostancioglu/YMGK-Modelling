from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np


N = 5 
F = 8  


def L96(x, t):
    d = np.zeros(N)

    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return d


x0 = F * np.ones(N)  
x0[0] += 0.01  
t = np.arange(0.0, 30.0, 0.01)

x = odeint(L96, x0, t)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(x[:, 0], x[:, 1], x[:, 2])
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
plt.show()