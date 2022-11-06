import matplotlib.pyplot as plt
import numpy as np

from oscisindy import data_generation

# generate data from a driven oscillator
timestep = 1e-2
time = np.arange(0., 10.0, timestep)


def driving_force(t, a, b, omega):
    """user defined force function"""
    if t > 4.:
        return b
    return a * np.sin(omega * 2 * np.pi * t) + b


signal = data_generation.integrate(time,
                                   data_generation.driven_oscillator,
                                   0.2,
                                   2. * np.pi * 2.)[:, 0]
driven_signal = data_generation.integrate(time,
                                          data_generation.driven_oscillator,
                                          0.2,
                                          2. * np.pi * 2.,
                                          control=(driving_force, (20, 3, 0.3)))[:, 0]

plt.figure()
plt.plot(time, signal, label='harmonic')
plt.plot(time, driven_signal, label='driven')
plt.legend()
plt.tight_layout()
plt.show()
