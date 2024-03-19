import os
import sys

import matplotlib.pyplot as plt
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
learning_trials = 20

test_time = reaching_time * 2
percent = 0.25

do_animate = True
# initialize arm model
moving_arm = 'right'
init_thetas = np.array((20, 20))

# define input
end_effectors = (np.array((-110, 200)), np.array((110, 200)))
input_in = [np.array((0, 0))]

target_in_motor_1 = [np.array((0, 0))]
target_in_motor_2 = [np.array((0, 0))]
target_in_trajectory = [np.array((0, 0))]

input_sparse = np.eye(len(end_effectors))
arms = PlanarArms(init_angles_right=init_thetas, init_angles_left=init_thetas, radians=False)

# learning trials
for learning_trial in range(learning_trials):
    for i, end_effector in enumerate(end_effectors):
        arms.reset_all()
        arms.move_to_position_and_return_to_init(moving_arm, end_effector, num_iterations=int(reaching_time))

        input_in += [input_sparse[i]] + [np.array((0, 0))] * int(reaching_time-1)
        target_in_trajectory += arms.trajectory_thetas_right
        if i == 0:
            target_in_motor_1 += arms.trajectory_thetas_right
            target_in_motor_2 += [np.array((0, 0))] * int(reaching_time)
        elif i == 1:
            target_in_motor_2 += arms.trajectory_thetas_right
            target_in_motor_1 += [np.array((0, 0))] * int(reaching_time)

length_learning = len(target_in_trajectory)
# testing trials
for i, end_effector in enumerate(end_effectors):
    arms.reset_all()
    arms.move_to_position_and_return_to_init(moving_arm, end_effector, num_iterations=int(reaching_time))
    if i == 0:
        target_in_motor_1 += arms.trajectory_thetas_right
    elif i == 1:
        target_in_motor_2 += arms.trajectory_thetas_right

target_in_motor_2[length_learning:int(percent*reaching_time)] = np.array((0, 0))
target_in_motor_1[(length_learning+int(percent*reaching_time)):] = np.array((0, 0))

arms.reset_all()
arms.change_to_position_in_trajectory_return_to_init(moving_arm, end_effectors, reaching_time,
                                                     break_at=int(percent*reaching_time))
target_in_trajectory += arms.trajectory_thetas_right

# save results in...
results_folder = f'results/kinematics_run_{simID}/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

ann.report(results_folder + "report.md")

# compile network
compile_folder = 'annarchy/'
if not os.path.exists(compile_folder):
    os.makedirs(compile_folder)
ann.compile(compile_folder + f'kinematics_run_{simID}')


# init pop monitor
mon_populations = [input_pop, reservoir, target_motor_plan_1, target_motor_plan_2, target_trajectory,
                   output_motor_plan_1, output_motor_plan_1, output_motor_plan_1,
                   output_motor_plan_2, output_motor_plan_2, output_motor_plan_2,
                   output_trajectory, output_trajectory, output_trajectory]

mon_variables = ['r', 'r', 'r', 'r',  'r',
                 'r', 'r_mean', 'noise',
                 'r', 'r_mean', 'noise',
                 'r', 'r_mean', 'noise']

rates = PopMonitor(populations=mon_populations, variables=mon_variables, sampling_rate=2.0)

weights = PopMonitor([res_output_motor_plan_1, res_output_motor_plan_2, res_output_trajectory],
                     variables=['w', 'w', 'w'], sampling_rate=50.0)

rates.start()
weights.start()

# learning
@ann.every(period=1.0)
def set_inputs(n):
    # Set inputs to the network
    input_pop.baseline = input_in[n]
    target_motor_plan_1.baseline = target_in_motor_1[n]
    target_motor_plan_2.baseline = target_in_motor_2[n]
    target_trajectory.baseline = target_in_trajectory[n]


ann.enable_learning()
ann.enable_callbacks()
ann.simulate(len(target_in_trajectory))

rates.save(folder=results_folder)
weights.save(folder=results_folder)

# disable callbacks
ann.disable_callbacks()

# testing condition
testing_rates = PopMonitor(populations=[input_pop, output_motor_plan_1, output_motor_plan_2, output_trajectory],
                           sampling_rate=1.0)
testing_rates.start()

# set targets to zero and deactivate noise in trajectory output
target_motor_plan_1.baseline = 0
target_motor_plan_2.baseline = 0
# disable noise fb
target_motor_plan_1.test = 1
target_motor_plan_2.test = 1

target_trajectory.baseline = 0
output_trajectory.test = 1

input_pop.baseline = [1, 0]

ann.step()

input_pop.baseline = 0
ann.simulate(percent * reaching_time)

input_pop.baseline = [0, 1]
ann.step()

input_pop.baseline = 0
ann.simulate(test_time)

res = testing_rates.get()
testing_rates.stop()

np.save(results_folder + 'test.npz', res, allow_pickle=True)

if do_animate:
    arms.set_trajectory(arm=moving_arm, trajectory=res['r_output_trajectory'])
    arms.plot_trajectory(points=end_effectors)
