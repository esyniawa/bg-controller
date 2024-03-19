import ANNarchy as ann


# FORCE learning rules
FORCE_reservoir_model = ann.Neuron(
    parameters="""
    tau = 50.0 : population
    phi = 0.0 : population
    chaos_factor = 1.0 : population
    """,
    equations="""
    noise = phi * Uniform(-1.0, 1.0)
    x += dt*(chaos_factor * sum(rec) + sum(fb) + sum(in) - x)/tau
    r = tanh(x) + noise
    """
)

FORCE_output_model = ann.Neuron(
    parameters="""
    baseline = 0.0 : population
    phi = 0.0 : population
    """,
    equations="""
    # input from reservoir (z)
    noise = phi * Uniform(-1.0, 1.0)

    r = sum(in) + baseline + noise

    e_minus = r - sum(target)
    e_plus = sum(in) - sum(target)
    """
)

FORCE_delta_learning_rule = ann.Synapse(
    parameters="""
        tau = 10. : projection
        eta = 0.005 : projection
        gamma = 2./3. : projection
        """,
    equations="""
        eta += eta/tau * (- eta + exp(gamma)/tau)
        delta_w = eta * post.e_minus * pre.r
        w -= delta_w
        """
)
