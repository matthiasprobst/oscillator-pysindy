"""
Generates HDF5 files of a driven damped harmonic oscillator in the data/ directory


from https://scientific-python.readthedocs.io/en/latest/notebooks_rst/3_Ordinary_Differential_Equations/02_Examples/Harmonic_Oscillator.html
"""
import pathlib
from typing import Callable, Union, Tuple, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

data_dir = pathlib.Path(__file__).parent / 'data'
data_dir.mkdir(parents=True, exist_ok=True)


def free_oscillator(X, t, zeta0, omega0):
    """
    Free Harmonic Oscillator ODE
    """
    x, dotx = X
    ddotx = -2 * zeta0 * omega0 * dotx - omega0 ** 2 * x
    return [dotx, ddotx]


def driven_oscillator(X: Union[List[np.ndarray], Tuple[np.ndarray, np.ndarray]],
                      t: np.ndarray,
                      contol_function: Callable,
                      *args):
    """
    Driven Harmonic Oscillator ODE
    control is a function that takes t (time) as an input

    *args is passed to the control function
    """
    zeta0 = args[0]
    omega0 = args[1]
    if contol_function is None:
        return free_oscillator(X, t, zeta0=zeta0, omega0=omega0)
    x, dotx = X
    ddotx = -2 * zeta0 * omega0 * dotx - omega0 ** 2 * x + contol_function(t, *args[2:])
    return [dotx, ddotx]


class Integrator:
    def __init__(self, control_function: Callable):
        self._control_function = control_function
        self.f = driven_oscillator

    def __call__(self, t, *params):
        return integrate(t, self.f, self._control_function, *params)[:, 0]


def integrate(t: np.ndarray,
              f: Callable,
              contol_function: Callable,
              *params):
    """
    Update function.
    """
    X0 = [1., 0.]
    return odeint(f,
                  X0,
                  t,
                  args=(contol_function, *params))


def control(t, a: float, omega: float, phi0, offset):
    return a * np.sin(omega * t - phi0) + offset


def write_hdf5_testdata(plot: bool = True):
    """main function generating the HDF5 data files"""
    dt = 1e-2
    t = np.arange(0., 10.0, dt)

    signal = integrate(t, driven_oscillator, 0.2, 2. * np.pi * 2.)[:, 0]
    driven_signal = integrate(t, driven_oscillator, 0.2, 2. * np.pi * 2., control=(control,
                                                                                   (0, 10, np.pi / 2, 20)))[:, 0]

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
