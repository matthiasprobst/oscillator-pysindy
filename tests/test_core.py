import h5py
import matplotlib.pyplot as plt
import numpy as np

from oscisindy import PySINDyData
from oscisindy.data_generation import write_hdf5_testdata, data_dir


def test():
    """main method"""
    write_hdf5_testdata(plot=False)
    with h5py.File(data_dir / 'test_data.hdf') as h5:
        # Init Data interfacing class and fit the model:
        data = PySINDyData(h5['free/time'][0:100], {'x': h5['free/signal'][0:100]}, 0.001, 1000)

        ptime = np.arange(0, 3 * h5['free/time'][-1], h5['free/time'][1] - h5['free/time'][0])
        data.predict(ptime)

        plt.figure()
        plt.plot(h5['free/time'][:], h5['free/signal'][:], label='original')
        data.plot()
        plt.legend()
        # plt.show()

        # driven:
        data = PySINDyData(h5['driven/time'][0:100], {'x': h5['driven/signal'][0:100]}, 0.001, 1000)

        ptime = np.arange(0, 3 * h5['driven/time'][-1], h5['driven/time'][1] - h5['driven/time'][0])
        data.predict(ptime)

        plt.figure()
        plt.plot(h5['driven/time'][:], h5['driven/signal'][:], label='original')
        data.plot()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    test()
