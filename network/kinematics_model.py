import numpy as np
import ANNarchy as ann

from .definitions import *

# model params
connect_prop = 0.1  # connection probability in reservoir
N_neurons_reservoir = 1000
N_neurons_output = 2  # [theta shoulder / theta elbow]

fb_strength = 1.0
in_strength = 1.0
target_strength = 1.0

# Input population
input_pop = ann.Population(geometry=N_neurons_output, neuron=input_neuron_dynamic, name='input_pop')

# Target population
target_pop_input = ann.Population(geometry=N_neurons_output, neuron=target_neuron, name='target_memory')
target_pop_trajectory = ann.Population(geometry=N_neurons_output, neuron=target_neuron, name='target_trajectory')

# Built reservoir
reservoir = ann.Population(geometry=N_neurons_reservoir, neuron=reservoir_model, name='reservoir_pop')
reservoir.chaos_factor = 1.2
reservoir.tau = 50.
# laterals
recurrent_res = ann.Projection(pre=reservoir, post=reservoir, target='rec')
recurrent_res.connect_fixed_probability(probability=connect_prop,
                                        weights=ann.Normal(0, np.sqrt(1/(connect_prop * N_neurons_reservoir))),
                                        allow_self_connections=True)

# output population
output_memory = ann.Population(geometry=N_neurons_output, neuron=output_model, name='output_memory')
output_trajectory = ann.Population(geometry=N_neurons_output, neuron=output_model, name='output_trajectory')


# reservoir -> output
res_output_memory = ann.Projection(pre=reservoir, post=output_memory,
                                   target='in',
                                   synapse=EH_learning_rule,
                                   name='eh_memory')
res_output_memory.connect_all_to_all(weights=0.0)  # set it to a very small value

res_output_trajectory = ann.Projection(pre=reservoir, post=output_trajectory,
                                       target='in',
                                       synapse=EH_learning_rule,
                                       name='eh_trajectory')
res_output_trajectory.connect_all_to_all(weights=0.0)  # set it to a very small value

# target -> output
target_output_memory = ann.Projection(pre=target_pop_input, post=output_memory, target='tar')
target_output_memory.connect_one_to_one(weights=target_strength)

target_output_trajectory = ann.Projection(pre=target_pop_trajectory, post=output_trajectory, target='tar')
target_output_trajectory.connect_one_to_one(weights=target_strength)

# feedback output -> reservoir
output_reservoir_proj = ann.Projection(pre=output_memory, post=reservoir, target='fb')
output_reservoir_proj.connect_all_to_all(weights=ann.Uniform(-fb_strength, fb_strength))

# input -> reservoir
input_reservoir_proj = ann.Projection(pre=input_pop, post=reservoir, target='in')
input_reservoir_proj.connect_all_to_all(weights=ann.Uniform(-in_strength, in_strength))
