import numpy as np
import scipy as sp

def simulate_closed_loop(N, mpc_solver, simulator, meas_func, estimator, params, W=None, V=None, w0=None, constr_state=None):
    
    # noise samples
    if W is None:
        W = np.random.randn(params["nw"], N)
    if V is None:
        V = np.random.randn(params["nv"], N)
    if w0 is None:
        w0 = np.random.randn(params["nw"])

    P0 = params["P0bar"]
    if type(P0) is not np.ndarray:
        P0 = np.array([[P0]])

    # sample initial system state
    X_true = np.nan * np.ones((params["nx"], N+1))
    X_true[:,0] = params["x0bar"].squeeze()
    X_true[:,0] += sp.linalg.sqrtm(P0) @ w0

    # applied control
    U = np.nan * np.ones((params["nu"], N))

    # estimated system state
    X_hat = np.nan * np.ones((params["nx"], N+1))
    X_hat[:,0] = params["x0bar"].squeeze()
    P_hat = [ P0 ]

    for k in range(N):
        # solve mpc
        mpc_solver.set_value_x0(X_hat[:,k])
        if mpc_solver.uncertainty_aware:
            mpc_solver.set_value_P0(P_hat[k])
        success = mpc_solver.solve()
        if not success:
            print("failed at timestep", k)
            return X_true[:,:k+1], U[:,:k], X_hat[:,:k+1], P_hat, 'solver_failed'
        U[:,k] = mpc_solver.get_value_u0()

        # system step
        X_true[:,k+1] = simulator(X_true[:,k], U[:,k], W[:,k], 0).full().squeeze()
        # if a state constraint is given, check for violation and stop simulation if violated
        if constr_state is not None:
            if not constr_state(X_true[:,k+1]) <= 0:
                return X_true[:,:k+1], U[:,:k], X_hat[:,:k+1], P_hat, 'constr_state_violated'

        # update state estimate
        # measurement
        y_meas = meas_func(X_true[:, k+1], V[:,k], 0)
        x_upd, P_upd = estimator( X_hat[:,k], U[:,k], y_meas, P_hat[-1], 0 )
        if type(x_upd) is not np.ndarray:
            x_upd = x_upd.full()
        X_hat[:,k+1] = x_upd.squeeze()
        if type(P_upd) is not np.ndarray:
            P_upd = P_upd.full()
        P_hat.append(P_upd)

        if mpc_solver.uncertainty_aware:
            mpc_solver.set_value_P0(P_hat[-1])
        mpc_solver.set_initial_guess_by_shifting()

    return X_true, U, X_hat, P_hat, 'success'
    