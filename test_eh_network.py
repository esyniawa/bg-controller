import os
import sys
import numpy as np

simID = int(sys.argv[1])
np.random.seed(simID)

import ANNarchy as ann
ann.setup(method='rk4', num_threads=1)

# import from scripts
from network.model import *
from monitoring import PopMonitor, ConMonitor

learning_time = 20. * 1000.  # 50 s
test_time = 20. * 1000.

# save results in...
results_folder = f'results/run_{simID}/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# compile network
compile_folder = 'annarchy/'
if not os.path.exists(compile_folder):
    os.makedirs(compile_folder)
ann.compile(compile_folder + f'run_{simID}')


# init pop monitor
rates = PopMonitor([reservoir, target_pop, output_pop, output_pop, output_pop, output_pop,
                    output_pop, output_pop],
                   variables=['r', 'r', 'r', 'r_mean', 'p', 'p_mean', 'm', 'noise'],
                   sampling_rate=1.0)

weights = PopMonitor([res_output_proj], variables=['w'], sampling_rate=20.0)

rates.start()
weights.start()

# init
ann.disable_learning()
ann.simulate(20.)

# learning
ann.set_time(0, net_id=0)  # to align target function with training and testing phase
ann.enable_learning()
ann.simulate(learning_time)

# testing condition
ann.disable_learning()
output_pop.test = 1

ann.simulate(test_time)

# save monitors
rates.stop()
rates.save(results_folder)

weights.stop()
weights.save(results_folder)
