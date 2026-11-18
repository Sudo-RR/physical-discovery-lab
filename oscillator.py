import numpy as np
def TimeEvolution(q0, p0, m, w, t):
    q0 = np.atleast_1d(np.asarray(q0, dtype=np.float64))
    p0 = np.atleast_1d(np.asarray(p0, dtype=np.float64))
    t = np.asarray(t, dtype=np.float64)
    # Valid Positions
    q0_axes = q0.ndim
    q0_size = q0.size
    q0_isvalid = np.isfinite(q0).all() and (q0_axes == 1) and (q0_size > 0)

    # Valid Positions
    p0_axes = p0.ndim
    p0_size = p0.size
    p0_isvalid = np.isfinite(p0).all() and (p0_axes == 1) and (p0_size > 0)
        
    # Valid Positions
    t_axes = t.ndim
    t_size = t.size
    t_isvalid = np.isfinite(t).all() and (t_axes == 1) and (t_size > 0)

    # Size Matching
    q0p0_isvalid = q0.shape == p0.shape
    
    if not(q0_isvalid):
        raise ValueError("Input Position is invalid")
    if not(p0_isvalid):
        raise ValueError("Input Momentum is invalid")
    if not(t_isvalid):
        raise ValueError("Input Time is invalid")
    if not(q0p0_isvalid):
        raise ValueError("Input Trajectories are invalid")

    if not np.isfinite(m) or m <= 0:
        raise ValueError("Mass must be finite and positive")
    if not np.isfinite(w) or w <= 0:
        raise ValueError("Angular frequency must be finite and positive")
    
    q0 = np.atleast_1d(np.asarray(q0, dtype=np.float64))
    p0 = np.atleast_1d(np.asarray(p0, dtype=np.float64))
    initial = np.array([q0,p0])#[:,None]
    M00 = np.cos(w * t)
    M01 = np.sin(w * t) / (m * w)
    M10 = - m * w * np.sin(w*t)
    M11 = np.cos(w * t)
    M = np.array([[M00,M01],[M10,M11]])
    M = np.moveaxis(M, 2,0)
    output = M @ initial
    return np.moveaxis(output,2,0)

