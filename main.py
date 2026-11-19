import numpy as np
import argparse
from generate_dataset import *
from oscillator import * 
from test import *

def main(config, dataset_path, test_output_path, validation_output_path, training_output_path):
    import json
    # Load configuration
    with open(config, "r") as f:
        config = json.load(f)

    # Generate dataset
    q0_train, p0_train, q0_validation, p0_validation, q0_test, p0_test, m, w, t = generate_dataset(config)

    # Save dataset to .npz file
    np.savez(dataset_path, 
             q0_train=q0_train, p0_train=p0_train,
             q0_validation=q0_validation, p0_validation=p0_validation,
             q0_test=q0_test, p0_test=p0_test)
    train_output = TimeEvolution(q0_train, p0_train, m, w, t)
    test_output = TimeEvolution(q0_test, p0_test, m, w, t)
    validation_output = TimeEvolution(q0_validation, p0_validation, m, w, t)
    print(f"Train Output Shape: {train_output.shape}")
    np.savez(training_output_path, train_output=train_output)
    np.savez(test_output_path, test_output=test_output)
    np.savez(validation_output_path, validation_output=validation_output)

    # Calculate errors
    E_initial_train = Energy(q0_train, p0_train, m, w)
    E_train = Energy(train_output[:,:,0], train_output[:,:,1], m, w)
    E_initial_validation = Energy(q0_validation, p0_validation, m, w)
    E_validation = Energy(validation_output[:,:,0], validation_output[:,:,1], m, w)
    E_initial_test = Energy(q0_test, p0_test, m, w)
    E_test = Energy(test_output[:,:,0], test_output[:,:,1], m, w)
    abs_error_train, relative_residual_train = Error(E_initial_train, E_train)
    abs_error_validation, relative_residual_validation = Error(E_initial_validation, E_validation)
    abs_error_test, relative_residual_test = Error(E_initial_test, E_test)

    max_abs_error_train = np.max(abs_error_train)
    max_relative_residual_train = np.max(relative_residual_train)
    max_abs_error_validation = np.max(abs_error_validation)
    max_relative_residual_validation = np.max(relative_residual_validation)
    max_abs_error_test = np.max(abs_error_test)
    max_relative_residual_test = np.max(relative_residual_test)

    print("Training Set Errors:")
    print("----------------------------------------------------------------------------")
    print(f"Max Absolute Error (Train): {max_abs_error_train}")
    print(f"Max Relative Residual (Train): {max_relative_residual_train}")

    print("Validation Set Errors:")
    print("----------------------------------------------------------------------------")
    print(f"Max Absolute Error (Validation): {max_abs_error_validation}")
    print(f"Max Relative Residual (Validation): {max_relative_residual_validation}")

    print("Test Set Errors:")
    print("----------------------------------------------------------------------------")
    print(f"Max Absolute Error (Test): {max_abs_error_test}")
    print(f"Max Relative Residual (Test): {max_relative_residual_test}")

    return train_output,validation_output,test_output, max_abs_error_train, max_relative_residual_train, max_abs_error_validation, max_relative_residual_validation, max_abs_error_test, max_relative_residual_test

import sys
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration JSON file.")
    parser.add_argument("--dataset-path", required=True, type=str, help="Path to save the generated dataset (.npz file).")
    parser.add_argument("--test-output-path", required=True, type=str, help="Path to save the test output (.npz file).")
    parser.add_argument("--validation-output-path", required=True, type=str, help="Path to save the validation output (.npz file).")
    parser.add_argument("--training-output-path", required=True, type=str, help="Path to save the training output (.npz file).")
    args = parser.parse_args()
    main(args.config, args.dataset_path, args.test_output_path, args.validation_output_path, args.training_output_path)