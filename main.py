import numpy as np

from generate_dataset import *
from oscillator import * 


def main(config):
    import json
    # Load configuration
    with open(config, "r") as f:
        config = json.load(f)

    # Generate dataset
    q0_train, p0_train, q0_validation, p0_validation, q0_test, p0_test, m, w, t = generate_dataset(config)

    # Save dataset to .npz file
    np.savez("dataset.npz", 
             q0_train=q0_train, p0_train=p0_train,
             q0_validation=q0_validation, p0_validation=p0_validation,
             q0_test=q0_test, p0_test=p0_test)
    train_output = TimeEvolution(q0_train, p0_train, m, w, t)
    test_output = TimeEvolution(q0_test, p0_test, m, w, t)
    validation_output = TimeEvolution(q0_validation, p0_validation, m, w, t)
    print(f"Train Output Shape: {train_output.shape}")
    np.savez("training_output.npz", train_output=train_output)
    np.savez("test_output.npz", test_output=test_output)
    np.savez("validation_output.npz", validation_output=validation_output)

    return train_output,validation_output,test_output

import sys
if __name__ == "__main__":
    main(sys.argv[1])