import casadi as ca
import numpy as np

def createEKFstepfunc(dyn_func, meas_func, proc_var=None, meas_var=None):
    """
    Create a CasADi function that performs one step of the extended Kalman filter.
    In:
        dyn_func:   dynamics of the form x_next = dyn_func(x, u, w, p)
        meas_func:  measurement function of the form y = meas_func(x, v, p)
        proc_var:   process noise covariance matrix. If None (default), it is assumed to be Identity.
        meas_var:   measurement noise covariance matrix. If None (default), it is assumed to be Identity.
    """

    nx = dyn_func.size1_in(0)
    nu = dyn_func.size1_in(1)
    nw = dyn_func.size1_in(2)
    npar = dyn_func.size1_in(3)
    ny = meas_func.size1_out(0)
    nv = meas_func.size1_in(1)

    if proc_var is None:
        proc_var = np.eye(nw)
    if meas_var is None:
        meas_var = np.eye(nv)

    # linearize dynamics
    x = ca.SX.sym('x', nx)
    u = ca.SX.sym('u', nu)
    w = ca.SX.sym('w', nw)
    p = ca.SX.sym('p', npar)
    v = ca.SX.sym('v', nv)
    y_meas = ca.SX.sym('y', ny)     # realized measurement
    P = ca.SX.sym('P', nx, nx)      # state estimate covariance

    # prediction step
    xp = dyn_func(x, u, w, p)
    A     = ca.substitute(ca.jacobian(xp, x), w, 0)
    Gamma = ca.substitute(ca.jacobian(xp, w), w, 0)
    x_pred = ca.substitute(xp, w, 0)
    P_pred = A @ P @ A.T + Gamma @ proc_var @ Gamma.T

    # update step
    y = meas_func(x, v, p)
    C = ca.substitute(ca.substitute(ca.jacobian(y, x), x, x_pred), v, 0)
    D = ca.substitute(ca.substitute(ca.jacobian(y, v), x, x_pred), v, 0)
    y_pred = ca.substitute(ca.substitute(y, x, x_pred), v, 0)
    S = C @ P_pred @ C.T + D @ meas_var @ D.T

    # K S = Ppred C.T <=> S.T K.T = C Ppred.T   (and Ppred.T = Ppred)
    K =  ca.solve(S.T, C @ P_pred ).T

    x_upd = x_pred + K @ ( y_meas - y_pred )
    P_upd = (ca.SX_eye(nx) - K @ C ) @ P_pred

    return ca.Function('EKF_step', [x, u, y_meas, P, p], [x_upd, P_upd])
