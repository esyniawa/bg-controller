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

do_animate = True
make_report = False


def make_inputs(
    reaching_time: int | float = 4. * 1000.,
    learning_trials: int = 4,
    percent: float = 0.25,
    moving_arm='right',
    init_thetas=np.array((20, 20)),
    end_effectors=(np.array((-110, 200)), np.array((110, 200)))
):

    test_time = reaching_time
    arms = PlanarArms(init_angles_right=init_thetas, init_angles_left=init_thetas, radians=False)

    in_input = [np.array((0, 0))]
    in_target_motor_1 = [np.array((0, 0))]
    in_target_motor_2 = [np.array((0, 0))]
    in_target_trajectory = [np.array((0, 0))]
    in_test = [0] * int(2 * learning_trials * reaching_time + 1) + [1] * int(test_time)

    # learning trials
    for learning_trial in range(learning_trials):
        for i, end_effector in enumerate(end_effectors):
            arms.reset_all()
            arms.move_to_position_and_return_to_init(moving_arm, end_effector, num_iterations=int(reaching_time))

            in_input += [np.eye(2)[i]] + [np.array((0, 0))] * int(reaching_time-1)
            in_target_trajectory += arms.trajectory_thetas_right
            if i == 0:
                in_target_motor_1 += arms.trajectory_thetas_right
                in_target_motor_2 += [np.array((0, 0))] * int(reaching_time)
            elif i == 1:
                in_target_motor_2 += arms.trajectory_thetas_right
                in_target_motor_1 += [np.array((0, 0))] * int(reaching_time)

    length_learning = len(in_target_trajectory)

    # testing trials
    in_input += [np.array((0, 0))] * int(test_time)
    in_input[length_learning + 1] = np.array((1, 0))
    in_input[length_learning + int(percent*reaching_time)] = np.array((0, 1))

    for i, end_effector in enumerate(end_effectors):
        arms.reset_all()
        arms.move_to_position_and_return_to_init(moving_arm, end_effector, num_iterations=int(test_time))
        if i == 0:
            in_target_motor_1 += arms.trajectory_thetas_right
        elif i == 1:
            in_target_motor_2 += arms.trajectory_thetas_right


    in_target_motor_2[length_learning:(length_learning + int(percent*reaching_time))] = [np.array((0, 0))] * int(percent*reaching_time)
    in_target_motor_1[(length_learning+int(percent*reaching_time)):] = [np.array((0, 0))] * int((1-percent)*reaching_time)

    arms.reset_all()
    arms.change_to_position_in_trajectory_return_to_init(moving_arm, end_effectors, num_iterations=int(test_time),
                                                         break_at=int(percent*reaching_time))
    in_target_trajectory += arms.trajectory_thetas_right
    arms.reset_all()

    return in_input, in_test, in_target_motor_1, in_target_motor_2, in_target_trajectory


# init inṕuts
reaching_time = int(4000)
end_effectors=(np.array((-110, 200)), np.array((110, 200)))
init_joint_angles=np.array((20, 20))
moving_arm = 'right'
in_input, in_test, in_target_motor_1, in_target_motor_2, in_target_trajectory = make_inputs(reaching_time=reaching_time,
                                                                                            moving_arm=moving_arm,
                                                                                            init_thetas=init_joint_angles,
                                                                                            end_effectors=end_effectors)

# save results in...
results_folder = f'results/kinematics_run_{simID}/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

if make_report:
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


# define callback to set inputs
@ann.every(period=1.0)
def set_inputs(n):
    # Set inputs to the network
    input_pop.baseline = in_input[n]
    # output_trajectory.test = in_test[n]

    target_motor_plan_1.baseline = in_target_motor_1[n]
    target_motor_plan_2.baseline = in_target_motor_2[n]
    target_trajectory.baseline = in_target_trajectory[n]


ann.enable_learning()
ann.enable_callbacks()
ann.simulate(len(in_target_trajectory) - 1)

rates.save(folder=results_folder)
weights.save(folder=results_folder)

res = rates.load(folder=results_folder)
trajectory = res['r_output_trajectory'][-reaching_time:]


if do_animate:
    arms = PlanarArms(init_angles_left=init_joint_angles, init_angles_right=init_joint_angles, radians=False)
    arms.set_trajectory(arm=moving_arm, trajectory=trajectory)
    arms.plot_trajectory(points=end_effectors)

