from typing import List, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from oscisindy import data_generation


def driving_force(t, a, b, omega):
    """user defined force function"""
    return a * np.sin(omega * 2 * np.pi * t) + b


class OscillatorControl:

    def __init__(self, f: Callable, params: List[float]):
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
    timestep = 1e-2
    time = np.arange(0.1, 10.0, timestep)

    ctrl = OscillatorControl(driving_force, (10, 20, 1))

    true_osci = Oscillator(control=ctrl, params=(0.1, 8))

    guessed_ctrl = OscillatorControl(driving_force, (3, 5, 4))
    guess_osci = Oscillator(control=guessed_ctrl, params=(1, 6))
    guess_osci.guess(time, true_osci.signal(time), method=None)  # 'trf, dogbox')

    plt.figure()
    plt.plot(time, true_osci.signal(time), label='true')
    plt.plot(time, guess_osci.signal(time), label='guess')
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fit_osci()
