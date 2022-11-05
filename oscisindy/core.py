import matplotlib.pyplot as plt
import numpy as np

import pysindy as ps

from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import STLSQ
from typing import Dict

THRESHOLD = 0.0001
MAX_ITERATIONS = 1000


def fit(x: np.ndarray, signals: Dict,
        ensemble: bool,
        th: float = THRESHOLD, max_iter: int = MAX_ITERATIONS):
    """Fit using pysindy

    x: time
    y1: List of signals
    """
    optimizer = STLSQ(threshold=th, max_iter=max_iter)
    # optimizer = TrappingSR3(threshold=THRESHOLD, max_iter=MAX_ITERATIONS)

    # differentiate the signal y(x)
    differentiation_method = FiniteDifference()

    signal_dots = [differentiation_method._differentiate(y, x) for y in signals.values()]

    # Get the model:
    data_to_stack = []
    feature_names = []
    for (sname, signal), signaldot in zip(signals.items(), signal_dots):
        feature_names.append(sname)
        feature_names.append(f'{sname}_dot')
        data_to_stack.append(signal)
        data_to_stack.append(signaldot)

    data = np.stack(data_to_stack)

    model = ps.SINDy(optimizer=optimizer,
                     differentiation_method=differentiation_method,
                     feature_names=feature_names,
                     discrete_time=False)
    model.fit(x=data.T, t=x, ensemble=ensemble)
    model.print()
    return model, {f'{sname}_dot': v for sname, v in zip(list(signals.keys()), signal_dots)}


def predict(model, time: np.ndarray, signals: Dict, derivatives: Dict):
    assert len(signals) == len(derivatives)

    initial_conditions = []
    for signal, derivative in zip(signals.values(), derivatives.values()):
        initial_conditions.append(signal[0])
        initial_conditions.append(derivative[0])
    prediction = model.simulate(initial_conditions, time)
    prediction_data = {}
    i = 0
    for signal, derivative in zip(signals.keys(), derivatives.keys()):
        prediction_data[signal] = prediction[:, i]
        i += 1
        prediction_data[derivative] = prediction[:, i]
        i += 1
    return prediction_data


class PySINDyData:

    def __init__(self, time, data: Dict, ensemble, th, max_iter):
        self.time = time
        self.data = data
        self.model, self.data_derivative_data = fit(self.time, data, ensemble, th, max_iter)

    def predict(self, time: np.ndarray):
        self.prediction_time = time
        self.prediction = predict(self.model, time, self.data, self.data_derivative_data)

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        for (n, d), p in zip(self.data.items(), self.prediction.values()):
            l = ax.plot(self.time, d, label=f'{n}', linestyle='-')
            ax.plot(self.prediction_time, p, label=f'{n} (pred)', linestyle='--', color=l[0].get_color())
        return ax
