import ANNarchy as ann
import numpy as np
from os import path, makedirs


class PopMonitor(object):
    def __init__(self, populations: tuple | list,
                 variables: tuple | list = None,
                 sampling_rate=2.0):

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

    def get(self):
        res = {}

        for i, monitor in enumerate(self.monitors):
            res[self.variables[i] + '_' + monitor.object.name] = monitor.get(self.variables[i])

        return res

    def save(self, folder):
        if not path.exists(folder):
            makedirs(folder)

        for i, monitor in enumerate(self.monitors):
            rec = monitor.get(self.variables[i])
            np.save(folder + self.variables[i] + '_' + monitor.object.name, rec)

    def load(self, folder):
        monitor_dict = {}

        for i, monitor in enumerate(self.monitors):
            monitor_dict[self.variables[i] + '_' + monitor.object.name] = np.load(
                folder + self.variables[i] + '_' + monitor.object.name + '.npy')

        return monitor_dict


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
        if not path.exists(folder):
            makedirs(folder)

        for con in self.connections:
            np.save(folder + 'w_' + con.name, self.weight_monitors[con.name])

    def load_cons(self, folder: str):
        con_dict = {}
        for con in self.connections:
            con_dict['w_' + con.name] = np.load(folder + 'w_' + con.name + '.npy')

    def reset(self):
        for con in self.connections:
            self.weight_monitors[con.name] = []
