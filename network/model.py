import numpy as np
import ANNarchy as ann

from .definitions import *

# model params
connect_prop = 0.1  # connection probability in reservoir
N_neurons_reservoir = 1000
N_neurons_output = 1

fb_strength = 1.0
target_strength = 1.0

# Target population
target_pop = ann.Population(geometry=N_neurons_output, neuron=test_target_neuron, name='target_pop')

# Built reservoir
reservoir = ann.Population(geometry=N_neurons_reservoir, neuron=reservoir_model, name='reservoir_pop')
recurrent_res = ann.Projection(pre=reservoir, post=reservoir, target='rec')
recurrent_res.connect_fixed_probability(probability=connect_prop,
                                        weights=ann.Normal(0, 1/(connect_prop * N_neurons_reservoir)))

# output population
output_pop = ann.Population(geometry=N_neurons_output, neuron=output_model_corr_noise, name='output_pop')

# reservoir -> output
res_output_proj = ann.Projection(pre=reservoir, post=output_pop,
                                 target='in',
                                 synapse=EH_learning_rule,
                                 name='eh_output')
res_output_proj.connect_all_to_all(weights=0.0)  # set it to a very small value

# target -> output
target_output_proj = ann.Projection(pre=target_pop, post=output_pop, target='tar')
target_output_proj.connect_one_to_one(weights=target_strength)

# feedback output -> reservoir
output_reservoir_proj = ann.Projection(pre=output_pop, post=reservoir, target='fb')
output_reservoir_proj.connect_all_to_all(weights=ann.Uniform(-fb_strength, fb_strength))
