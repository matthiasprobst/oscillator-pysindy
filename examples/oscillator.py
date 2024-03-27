from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from oscisindy import data_generation


def driving_force(t, a, b, omega, phi0):
    """user defined force function"""
    return a * np.sin(omega * 2 * np.pi * t - phi0) + b


def guessed_force(t, c):
    return c


class OscillatorControl:

    def __init__(self, f: Callable, params: Tuple):
        self.f = f
        self.params = params


class Oscillator:
    """Oscillator class"""

    def __init__(self, control: OscillatorControl, params: Tuple):
        self._control = control
        self._params = params

    @staticmethod
    def _integrate(time, control_function: Callable, *params):
        return data_generation.integrate(time,
                                         data_generation.driven_oscillator,
                                         control_function,
                                         *params)[:, 0]

    def signal(self, time: np.ndarray):
        """signal"""
        return self._integrate(time, self._control.f, *(*self._params, *self._control.params))

    def guess(self, time, signal, *args, **kwargs):
        """Guess parameters based on other signal"""
        Int = data_generation.Integrator(self._control.f)
        popt, pcov = optimize.curve_fit(Int,  # our function
                                        xdata=time,  # measured x values
                                        ydata=signal,  # measured y values
                                        p0=(*self._params,
                                            *self._control.params),
                                        *args,
                                        **kwargs)  # the initial guess for the two parameters
        print(f'Found parameters: {popt}')
        self._params = tuple(popt[0:2])
        self._control.params = tuple(popt[2:])


def fit_osci():
    # generate data from a driven oscillator
    Nt = 800
    timestep = 1e-2
    time = np.arange(0.1, 10.0, timestep)
    time_exp = np.arange(0.1, 10.0, timestep)

    true_ctrl = (10, 20, 2)
    guess_ctrl = (1, 1, 1)

    ctrl = OscillatorControl(driving_force, true_ctrl)
    true_osci = Oscillator(control=ctrl, params=(0.1, 8))

    guessed_ctrl = OscillatorControl(driving_force, guess_ctrl)
    guess_osci = Oscillator(control=guessed_ctrl, params=(1, 1))

    guess_osci.guess(time[0:Nt], true_osci.signal(time[0:Nt]), method=None)  # 'trf, dogbox')

    plt.figure()
    plt.plot(time_exp, guess_osci.signal(time_exp), label='prediction')
    plt.plot(time[:Nt], guess_osci.signal(time[:Nt]), label='fit')
    plt.plot(time, true_osci.signal(time), label='true')
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fit_osci()
