import os
import sys
import numpy as np

simID = int(sys.argv[1])
np.random.seed(simID)

import ANNarchy as ann
ann.setup(method='rk4')

# import from scripts
from network.model import *
from monitoring import PopMonitor, ConMonitor

learning_time = 10. * 1000.  # 10 s
test_time = 2000.

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

rates.start()

con_monitor = ConMonitor([res_output_proj])
con_monitor.extract_weights()

# init
ann.disable_learning()
ann.simulate(50.)

# learning
ann.enable_learning()
ann.simulate(learning_time)

# testing condition
ann.disable_learning()
output_pop.test = 1

ann.simulate(test_time)

# save monitors
rates.stop()
monitors = rates.save(results_folder)

con_monitor.extract_weights()
con_monitor.save_cons(results_folder)
