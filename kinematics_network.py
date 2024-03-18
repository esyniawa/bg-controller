import os
import sys
import numpy as np

simID = int(sys.argv[1])
np.random.seed(simID)

import ANNarchy as ann
ann.setup(method='rk4', num_threads=1)

# import from scripts
from network.kinematics_model import *
from monitoring import PopMonitor

from kinematics.planar_arms import PlanarArms

# parameters
reaching_time = 4. * 1000.  # 5 [s]
learning_trials = 5

test_time = reaching_time
percent = 0.25

do_animate = True
# initialize arm model
moving_arm = 'right'
init_thetas = np.array((20, 20))
end_effectors = (np.array((-110, 200)), np.array((110, 200)))
input_in = [np.array((0, 0))]
input_sparse = np.eye(len(end_effectors))
arms = PlanarArms(init_angles_right=init_thetas, init_angles_left=init_thetas, radians=False)

for learning_trial in range(learning_trials):
    for i, end_effector in enumerate(end_effectors):
        arms.move_to_position_and_return_to_init(moving_arm, end_effector, num_iterations=int(reaching_time))
        input_in += [input_sparse[i]]*int(reaching_time)

if moving_arm == 'right':
    target_in = arms.trajectory_thetas_right
else:
    target_in = arms.trajectory_thetas_left

# define input
# save results in...
results_folder = f'results/kinematics_run_{simID}/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# compile network
compile_folder = 'annarchy/'
if not os.path.exists(compile_folder):
    os.makedirs(compile_folder)
ann.compile(compile_folder + f'kinematics_run_{simID}')


# init pop monitor
mon_populations = [input_pop, reservoir, target_pop_input, target_pop_trajectory,
                   output_memory, output_memory, output_memory,
                   output_trajectory, output_trajectory, output_trajectory]
mon_variables = ['r', 'r', 'r', 'r',
                 'r', 'r_mean', 'noise',
                 'r', 'r_mean', 'noise']

rates = PopMonitor(populations=mon_populations, variables=mon_variables, sampling_rate=10.0)

weights = PopMonitor([res_output_memory, res_output_trajectory], variables=['w', 'w'], sampling_rate=50.0)

rates.start()
weights.start()

# init
ann.disable_learning()
ann.simulate(20.)


# learning
@ann.every(period=1.0)
def set_inputs(n):
    # Set inputs to the network
    input_pop.baseline = input_in[n]
    target_pop_input.baseline = input_in[n]
    target_pop_trajectory.baseline = target_in[n]


ann.enable_learning()
ann.simulate(len(target_in))

rates.save(folder=results_folder)
weights.save(folder=results_folder)

rates.stop()
weights.stop()

# disable callbacks
ann.disable_callbacks()
ann.disable_learning()
ann.simulate(20)

# testing condition
testing_rates = PopMonitor(populations=[input_pop, output_memory, output_trajectory], sampling_rate=1.0)
testing_rates.start()

target_pop_input.baseline = 0
target_pop_trajectory.baseline = 0

input_pop.baseline = [0, 1]

output_memory.test = 1
output_trajectory.test = 1

ann.simulate(test_time + 10)

res = testing_rates.get()
testing_rates.stop()

np.save(results_folder + 'test.npz', res, allow_pickle=True)

if do_animate:
    arms.set_trajectory(arm=moving_arm, trajectory=res['r_output_trajectory'])
    arms.plot_trajectory(points=end_effectors)
