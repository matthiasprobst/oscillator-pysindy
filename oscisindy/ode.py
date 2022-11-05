import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import Callable
import h5py
from .data_generation import data_dir

def fit_ode():
    pass


def f(x, t, A, omega, phi):
    """
    Your system of differential equations
    """
    # x' = x1
    # x1' = '1', 'x', 'x_dot', 'x^2', 'x x_dot', 'x_dot^2
    # the model equations
    f0 = x[1]

    f1 = 16.242 * 1 + -165.810 * x[0] + \
         -2.198 * x[1] + 13.009 * x[0] ** 2 + \
         2.564 * x[0] * x[1] + 0.333 * x[1] ** 2 + \
         A * np.sin(omega * t - phi)
    return [f0, f1]


def integrate(t, f: Callable, x0, A, omega, phi):
    """
    Update function.
    """
    return scipy.integrate.odeint(f, x0, t, args=(A, omega, phi))


def test():
    with h5py.File(data_dir / 'test_data.hdf') as h5:
        t = h5['driven/time'][:]
        signal = h5['driven/signal'][:]

    x0 = signal[0], (signal[1]-signal[0])/(t[1]-t[0])
    model = integrate(t, f, x0, 20, 10, np.pi / 2)

    plt.figure()
    plt.plot(t, model[:, 0], label='model')
    plt.plot(t, signal, label='true')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test()
