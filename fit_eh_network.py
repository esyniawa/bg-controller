import os
import sys
import numpy as np

from pybads.bads import BADS
from contextlib import contextmanager

simID = int(sys.argv[1])
np.random.seed(simID)

import ANNarchy as ann
ann.setup(method='rk4', num_threads=1)

# import from scripts
from network.model import *
from monitoring import PopMonitor


# supress standard output
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# For resetting weights
class InitialWeights(object):
    def __init__(self, connections, folder: str):
        self.connections = connections
        self.folder = folder

    def write_weights(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder + 'projections/')

        for con in self.connections:
            con.save(self.folder + 'projections/' + con.name + '.npz')

    def load_weights(self):
        for con in self.connections:
            con.load(self.folder + 'projections/' + con.name + '.npz')


# Compute the mean reward per trial
def fit_reservoir(folder: str,
                  init_eta=0.0005,
                  exploratory_noise=0.5,
                  chaos_res=1.5,
                  learning_time=20. * 1000.,  # 5 s
                  test_time=5. * 1000.,
                  do_test=True) -> None:

    my_params = (init_eta, exploratory_noise, chaos_res)

    # presentation times
    t_init = 20.  # in [ms]

    # compile reservoir
    compile_folder = f"annarchy/fit_seed_{simID}/"
    if not os.path.exists(compile_folder):
        os.makedirs(compile_folder)
    ann.compile(directory=compile_folder)

    # init reservoir weights
    init_w = InitialWeights([res_output_proj], folder=compile_folder)
    init_w.write_weights()

    # init monitors
    performance = PopMonitor([output_pop], variables=['p'], sampling_rate=1)

    def loss_function(res_params):

        param_eta, param_phi, param_lambda = res_params

        # set parameters in reservoir and synapses
        res_output_proj.eta_init = param_eta
        output_pop.phi = param_phi
        reservoir.chaos_factor = param_lambda

        # reset weights
        init_w.load_weights()

        # init
        ann.disable_learning()
        output_pop.test = 0
        ann.simulate(t_init)

        # learning
        ann.enable_learning()
        ann.simulate(learning_time)

        # testing condition
        ann.disable_learning()
        output_pop.test = 1

        performance.start()
        ann.simulate(test_time)

        tracking = performance.get()
        error = np.log(np.mean(-tracking['p_output_pop'])) + np.log(np.amax(-tracking['p_output_pop']))

        performance.stop()
        ann.reset(monitors=True)

        return error

    def test_function(res_params) -> None:

        monitors = PopMonitor(populations=[target_pop, output_pop],
                              variables=['r', 'r'],
                              sampling_rate=1.0)

        param_eta, param_phi, param_lambda = res_params

        # set parameters in reservoir and synapses
        res_output_proj.eta_init = param_eta
        output_pop.phi = param_phi
        reservoir.chaos_factor = param_lambda

        # reset weights
        init_w.load_weights()

        monitors.start()

        # init
        ann.disable_learning()
        output_pop.test = 0
        ann.simulate(t_init)

        # learning
        ann.enable_learning()
        ann.simulate(learning_time)

        # testing condition
        ann.disable_learning()
        output_pop.test = 1

        monitors.save(results_folder)
        monitors.stop()

    target = loss_function

    bads = BADS(target, np.array(my_params),
                lower_bounds=np.array((0.00000001, 0.00000001, 0.00000001)),
                upper_bounds=np.array((1., 3.5, 6.)))

    optimize_result = bads.optimize()
    fitted_params = optimize_result['x']

    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(folder + "fitted_params.npy", fitted_params)

    # test network with fitted params
    if do_test:
        test_function(fitted_params)


if __name__ == '__main__':

    results_folder = f"results/fit_{simID}/"

    with suppress_stdout():
        fit_reservoir(folder=results_folder)
