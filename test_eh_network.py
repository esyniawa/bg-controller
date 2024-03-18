import os
import sys
import numpy as np

simID = int(sys.argv[1])
np.random.seed(simID)

import ANNarchy as ann
ann.setup(method='rk4', num_threads=1)

import matplotlib.pyplot as plt

# import from scripts
from network.closed_loop_model import *
from monitoring import PopMonitor, ConMonitor

learning_time = 30. * 1000.  # 50 s
test_time = 5. * 1000.

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
mon_populations = [reservoir, target_pop, output_pop, output_pop, output_pop, output_pop, output_pop, output_pop]
mon_variables = ['r', 'r', 'r', 'r_mean', 'p', 'p_mean', 'm', 'noise']

rates = PopMonitor(populations=mon_populations, variables=mon_variables, sampling_rate=1.0)

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
ann.set_time(0, net_id=0)
ann.disable_learning()
output_pop.test = 1

ann.simulate(test_time)

# save monitors
rates.save(results_folder)
rates.stop()

weights.save(results_folder)
weights.stop()

fig, ax = plt.subplots(figsize=(40, 14))
monitors = rates.load(results_folder)

res_output = monitors['r_output_pop']
res_target = monitors['r_target_pop']
res_output_noiseless = monitors['r_output_pop'][:int(learning_time)] - monitors['noise_output_pop'][:int(learning_time)]
ax.plot(res_output, c="b", alpha=0.2)
ax.plot(res_output_noiseless, c="b")
ax.plot(res_target, c="r", alpha=0.4)

plt.savefig(results_folder + "learned_trajectory.pdf")
plt.close(fig)
