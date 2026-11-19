import numpy as np

def Energy(q, p, m, w):
    
    E = p**2 / (2*m) + 0.5*m*w**2*q**2  # shape (B, T)
    return E

def Error(E_initial,E):
    E_initial = np.atleast_1d(np.asarray(E_initial, dtype=np.float64))
    
    E_initial = E_initial[:,None]  # shape (B, T)

    E = np.atleast_1d(np.asarray(E, dtype=np.float64))

    abs_error = np.abs(E - E_initial)  # shape (B, T)

    nonzero_indices = E_initial[:, 0] > 0

    relative_residual = abs_error[nonzero_indices,:] / E_initial[nonzero_indices,:]

    return abs_error, relative_residual

