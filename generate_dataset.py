import numpy as np
import json

def generate_dataset(config):

    num_samples = config["num_trajectories"]

    rng = np.random.default_rng(config["seed"])

    A_min , A_max = config["amplitude_min"], config["amplitude_max"] #amplitude

    log_A = rng.uniform(np.log(A_min), np.log(A_max), size=num_samples)

    A = np.exp(log_A)

    phi_min , phi_max = config["phase_min"]*np.pi, config["phase_max"]*np.pi #phase
    
    phi = rng.uniform(phi_min, phi_max, size=num_samples) #phase

    m = config["mass"]

    w = config["omega"]

    t_min , t_max = 0 , config["num_periods"]*2*np.pi/w #times

    t = np.linspace(t_min, t_max, config["num_times"])

    q0 = A*np.cos(phi)

    p0 = -m*w*A*np.sin(phi)

    indices = rng.permutation(num_samples)

    train_indices = indices[:config["num_train"]]

    validation_indices = indices[config["num_train"]:config["num_train"]+config["num_validation"]]

    test_indices = indices[config["num_train"]+config["num_validation"]:]

    q0_train = q0[train_indices]
    p0_train = p0[train_indices]

    q0_validation = q0[validation_indices]
    p0_validation = p0[validation_indices]

    q0_test = q0[test_indices]
    p0_test = p0[test_indices]

    return q0_train, p0_train, q0_validation, p0_validation, q0_test, p0_test, m, w, t
