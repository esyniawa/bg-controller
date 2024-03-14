import ANNarchy as ann


reservoir_model = ann.Neuron(
    parameters="""
    tau = 50.0 : population
    phi = 0.05 : population
    l = 1.5 : population
    """,
    equations="""
    noise = phi * Uniform(-1.0, 1.0)
    x += dt*(l * sum(rec) + sum(fb) + sum(in) - x)/tau
    r = tanh(x) + noise
    """
)

output_model = ann.Neuron(
    parameters="""
    alpha_r = 0.8 : population
    alpha_p = 0.8 : population
    baseline = 0.0 : population
    phi = 0.5 : population
    test = 0 : population, bool
    """,
    equations="""
    # input from reservoir (z)
    noise = phi * Uniform(-1.0, 1.0)
    r_in = sum(in) + baseline
    r = if (test == 0):
            r_in + noise
        else:
            r_in

    r_mean = alpha_r * r_mean + (1 - alpha_r) * r

    # performance
    p = - power(r - sum(tar), 2)
    p_mean = alpha_p * p_mean + (1 - alpha_p) * p

    # modulatory signal
    m = if (p > p_mean):
            1.0
        else:
            0.0
    """
)

output_model_corr_noise = ann.Neuron(
    parameters="""
    alpha_r = 0.8 : population
    alpha_p = 0.8 : population
    alpha_noise = 0.6 : population
    baseline = 0.0 : population
    phi = 0.5 : population
    test = 0 : population, bool
    """,
    equations="""
    # temporally correlated exploration noise
    noise_new = phi * Uniform(-1.0, 1.0)
    noise = (1 - alpha_noise) * noise + alpha_noise * noise_new 

    # input from reservoir (z)
    r_in = sum(in) + baseline
    r = if (test == 0):
            r_in + noise
        else:
            r_in

    r_mean = alpha_r * r_mean + (1 - alpha_r) * r

    # performance
    p = - power(r - sum(tar), 2)
    p_mean = alpha_p * p_mean + (1 - alpha_p) * p

    # modulatory signal
    m = if (p > p_mean):
            1.0
        else:
            0.0
    """
)

target_neuron = ann.Neuron(
    parameters="""
    baseline = 0.0
    phi = 0.0 : population
    """,
    equations="""
    r = baseline + phi * Uniform(-1.0,1.0)
    """
)

test_target_neuron = ann.Neuron(
    parameters="""
    w1 = 0.5 : population
    w2 = 0.0 : population
    w3 = 0.35 : population

    f1 = 1/1000 : population
    f2 = 1/20 : population
    f3 = 1/500 : population
    """,
    equations="""
    r = w1 * sin(2 * pi * f1 * t) + w2 * sin(2 * pi * f2 * t) + w3 * sin(2 * pi * f3 * t) 
    """
)

EH_learning_rule = ann.Synapse(
    parameters="""
    eta_init = 0.005 
    decay = 10000.
    """,
    equations="""
    learning_rate = eta_init / (1 + t/decay)
    delta_w = learning_rate * (post.r - post.r_mean) * post.m * pre.r
    w += delta_w
    """
)
