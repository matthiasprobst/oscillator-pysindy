"""
Generates HDF5 files of a driven damped harmonic oscillator in the data/ directory


from https://scientific-python.readthedocs.io/en/latest/notebooks_rst/3_Ordinary_Differential_Equations/02_Examples/Harmonic_Oscillator.html
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import scipy
from typing import Callable

data_dir = pathlib.Path(__file__).parent / 'data'
data_dir.mkdir(parents=True, exist_ok=True)


def free_oscillator(X, t, zeta, omega0):
    """
    Free Harmonic Oscillator ODE
    """
    x, dotx = X
    ddotx = -2 * zeta * omega0 * dotx - omega0 ** 2 * x
    return [dotx, ddotx]


def driven_oscillator(X, t, zeta, omega0, control: Callable):
    """
    Driven Harmonic Oscillator ODE
    control is a function that takes t (time) as an input
    """
    if control is None:
        return free_oscillator(X, t, zeta, omega0)
    x, dotx = X
    ddotx = -2 * zeta * omega0 * dotx - omega0 ** 2 * x + control(t)
    return [dotx, ddotx]


def integrate(t, f: Callable, zeta=0.05, omega0=2. * np.pi, control=None):
    """
    Update function.
    """
    X0 = [1., 0.]
    return scipy.integrate.odeint(f, X0, t, args=(zeta, omega0, control))


def define_control(a: float, omega: float, phi0, offset) -> Callable:
    """Generates the control function that takes time as input only"""

    def control(t):
        return a * np.sin(omega * t - phi0) + offset

    return control


def write_hdf5_testdata(plot: bool = True):
    """main function generating the HDF5 data files"""
    dt = 1e-2
    t = np.arange(0., 10.0, dt)

    ctr = define_control(20, 10, np.pi / 2, offset=20)

    signal = integrate(t, driven_oscillator, 0.2, 2. * np.pi * 2.)[:, 0]
    driven_signal = integrate(t, driven_oscillator, 0.2, 2. * np.pi * 2., control=ctr)[:, 0]

    if plot:
        fig = plt.figure()
        plt.plot(t, signal, label='non-driven')
        plt.plot(t, driven_signal, label='driven')
        plt.legend()
        plt.ylim(-1., 1.)
        plt.xlabel("Time, $t$")
        plt.ylabel("Amplitude, $a$")
        plt.show()

    with h5py.File(data_dir / 'test_data.hdf', 'w') as h5:
        h5.create_dataset(name="free/signal", data=signal)
        h5.create_dataset(name="free/time", data=t)
        h5.create_dataset(name="driven/signal", data=driven_signal)
        h5.create_dataset(name="driven/time", data=t)


if __name__ == '__main__':
    write_hdf5_testdata()
