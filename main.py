import numpy as np
from generate_dataset import *
from oscillator import * 

if __name__ == "__main__":
    # Load configuration
    with open("configs/sho.json", "r") as f:
        config = json.load(f)

    # Generate dataset
    q0_train, p0_train, q0_validation, p0_validation, q0_test, p0_test, m, w, t = generate_dataset(config)

    # Save dataset to .npz file
    np.savez("dataset.npz", 
             q0_train=q0_train, p0_train=p0_train,
             q0_validation=q0_validation, p0_validation=p0_validation,
             q0_test=q0_test, p0_test=p0_test)
    train_output = TimeEvolution(q0_train, p0_train, m, w, t)
    print(f"Train Output Shape: {train_output.shape}")