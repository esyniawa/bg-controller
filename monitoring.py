import ANNarchy as ann
import matplotlib.pyplot as plt
import numpy as np
import os


class PopMonitor(object):
    def __init__(self, populations: tuple | list,
                 variables: tuple | list | None = None,
                 sampling_rate: float = 2.0):

        # define variables to track
        if variables is not None:
            assert len(populations) == len(variables), "The Arrays of populations and variables must have the same length"

            self.variables = variables
        else:
            self.variables = ['r'] * len(populations)

        # init monitors
        self.monitors = []

        for i, pop in enumerate(populations):
            self.monitors.append(ann.Monitor(pop, self.variables[i], period=sampling_rate, start=False))

    def start(self):
        for monitor in self.monitors:
            monitor.start()

    def stop(self):
        for monitor in self.monitors:
            monitor.pause()

    def resume(self):
        for monitor in self.monitors:
            monitor.resume()

    def get(self, delete: bool = True):
        res = {}

        for i, monitor in enumerate(self.monitors):
            res[self.variables[i] + '_' + monitor.object.name] = monitor.get(self.variables[i], keep=not delete)

        return res

    def save(self, folder, delete: bool = True):
        if not os.path.exists(folder):
            os.makedirs(folder)

        for i, monitor in enumerate(self.monitors):
            rec = monitor.get(self.variables[i], keep=not delete)
            np.save(folder + self.variables[i] + '_' + monitor.object.name, rec)

    def load(self, folder):
        monitor_dict = {}

        for i, monitor in enumerate(self.monitors):
            monitor_dict[self.variables[i] + '_' + monitor.object.name] = np.load(
                folder + self.variables[i] + '_' + monitor.object.name + '.npy')

        return monitor_dict

    @staticmethod
    def _2D_reshape(m: np.ndarray):
        shape = m.shape

        for i in range(m.ndim, 2, -1):
            new_shape = list(shape[:-1])
            new_shape[-1] = shape[-1] * shape[-2]
            shape = new_shape

        return m.reshape(shape)

    def plot_rates(self, plot_order: tuple[int, int],
                   fig_size: tuple[float, float] | list[float, float],
                   save_name: str = None):

        """
        PLots 2D populations rates.
        :param plot_order:
        :param fig_size:
        :param save_name:
        :return:
        """
        ncols, nrows = plot_order
        results = self.get(delete=False)

        fig = plt.figure(figsize=fig_size)
        for i, key in enumerate(results):
            if results[key].ndim > 2:
                results[key] = PopMonitor._2D_reshape(results[key])

            plt.subplot(nrows, ncols, i + 1)
            plt.plot(results[key])
            plt.title(self.monitors[i].object.name, loc='left')
            plt.ylabel('Activity')
            plt.xlabel(self.variables[i], loc='right')

        if save_name is None:
            plt.show()
        else:
            folder, _ = os.path.split(save_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            plt.savefig(save_name)
            plt.close(fig)

    def weight_difference(self, plot_order: tuple[int, int],
                          fig_size: tuple[float, float] | list[float, float],
                          save_name: str = None):
        """
        Plots weight difference with imshow. Weights are normally recorded in [time, pre_synapse, post_synapse].
        :param plot_order:
        :param fig_size:
        :param save_name:
        :return:
        """

        ncols, nrows = plot_order
        results = self.get(delete=False)

        fig = plt.figure(figsize=fig_size)
        for i, key in enumerate(results):
            plt.subplot(ncols, nrows, i + 1)

            difference = results[key][-1, :, :] - results[key][0, :, :]
            x_axis = np.arange(0, len(difference.T))
            plt.plot(difference.T)
            plt.xticks(x_axis, x_axis+1)
            plt.xlabel('pre-synaptic neuron', loc='right')
            plt.title(key + ' : Weight difference', loc='left')

        if save_name is None:
            plt.show()
        else:
            folder, _ = os.path.split(save_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            plt.savefig(save_name)
            plt.close(fig)


class ConMonitor(object):
    def __init__(self, connections):
        self.connections = connections
        self.weight_monitors = {}
        for con in connections:
            self.weight_monitors[con.name] = []

    def extract_weights(self):
        for con in self.connections:
            weights = np.array([dendrite.w for dendrite in con])
            self.weight_monitors[con.name].append(weights)

    def save_cons(self, folder: str):
        if not os.path.exists(folder):
            os.makedirs(folder)

        for con in self.connections:
            np.save(folder + 'w_' + con.name, self.weight_monitors[con.name])

    def load_cons(self, folder: str):
        con_dict = {}
        for con in self.connections:
            con_dict['w_' + con.name] = np.load(folder + 'w_' + con.name + '.npy')

        return con_dict

    def reset(self):
        for con in self.connections:
            self.weight_monitors[con.name] = []
