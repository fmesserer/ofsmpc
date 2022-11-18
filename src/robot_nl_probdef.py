import numpy as np
import casadi as ca

from ellipsoid_utils import vecToSymm
from utils import huberLoss

def ode(x, u, w, params):
    '''
    ode of system
    state x = [rx, ry, theta]
    control u = [ v, omega ]
    '''

    theta = x[2]
    vel = u[0]
    omg = u[1]

    dot_rx = vel * ca.cos(theta)
    dot_ry = vel * ca.sin(theta)
    dot_theta = omg
    xdot = ca.veccat(dot_rx, dot_ry, dot_theta)

    # noise scaling, assuming w is standard normal distributed
    noise_scale = params['process_std_scale']
    xdot += noise_scale @ w
    return xdot


def rk4_step(x, u, w, dt, params):
    """
    One step of RK4 integration of the ode defined above
    """
    k1       = ode(x,            u, w, params=params)
    k2       = ode(x + dt/2 * k1, u, w, params=params)
    k3       = ode(x + dt/2 * k2, u, w, params=params)
    k4       = ode(x + dt * k3,   u, w, params=params)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def get_dyn_discr(params):
    """
    create discrete time dynamics function of the ODE from above
    output is a casadi function of the form x_next = f(x, u, w, p)
    where x is the state, u is the control, w is the process noise, and p is a parameter
    """

    x = ca.SX.sym("x", params["nx"])
    u = ca.SX.sym("u", params["nu"])
    w = ca.SX.sym("w", params["nw"])
    p = ca.SX.sym("p", params["np"])
    dt = params["T"] / params["N"]
    xnext = rk4_step(x, u, w, dt, params)
    return ca.Function('F_step', [x, u, w, p], [xnext] )


def get_dyn_nom(params):
    """
    Create version of discretized dynamics with no noise argument (for nominal MPC solver)
    """
    x = ca.SX.sym("x", params["nx"])
    u = ca.SX.sym("u", params["nu"])
    p = ca.SX.sym("p", params["np"])
    dyn_discr = get_dyn_discr(params)
    return ca.Function('dyn_nom', [x, u, p], [dyn_discr(x, u, 0, p)])


def get_cost_stage(params):
    """
    create uncertainty aware stage cost function of the form l = l(x, u, Sigma_xu, sx, su, p)
    where x is the state, u is the control, Sigma_xu is the joint covariance of x and u,
    sx is a slack of the state constraint, su is a slack of the control constraint, and p is a parameter
    """

    x = ca.SX.sym("x", params["nx"])
    u = ca.SX.sym("u", params["nu"])
    sx = ca.SX.sym("sx", params["nsx"])
    su = ca.SX.sym("su", params["nsu"])
    p = ca.SX.sym("p", params["np"])
    n_sig = params["nx"] + params["nu"]
    Sigma_vec = ca.SX.sym("Sigma", n_sig * (n_sig + 1) // 2)
    Sigma = vecToSymm(Sigma_vec, n_sig)
    Sigma_u = Sigma[params["nx"]:, params["nx"]:]

    l = x[0]
    R = 1e-6 * ca.diag([1, 1])
    l += u.T @ R @ u
    l += ca.trace(Sigma_u @ R)

    return ca.Function('cost_stage', [x, u, Sigma_vec, sx, su, p], [l])


def get_cost_terminal(params):
    """
    Create terminal cost function of the form l = l(x, Sigma_x, sx, sxN, p)
    where x is the state, Sigma_x is the covariance of x, sx is a slack of the state constraint,
    sxN is a slack of the terminal state constraint, and p is a parameter
    """

    x = ca.SX.sym("x", params["nx"])
    sx = ca.SX.sym("sx", params["nsx"])
    sxN = ca.SX.sym("sxN", params["nsxN"])
    p = ca.SX.sym("p", params["np"])
    n_sig = params["nx"]
    Sigma_vec = ca.SX.sym("Sigma", n_sig * (n_sig + 1) // 2)
    Sigma = vecToSymm(Sigma_vec, n_sig)

    l = x[0]
    return ca.Function('cost_terminal', [x, Sigma_vec, sx,  sxN, p], [l])


def get_output_func(params):
    """
    output / measurement function: y = g(x, v)
    """
    x = ca.SX.sym('x', params["nx"])
    v = ca.SX.sym('v', params["nv"])
    p = ca.SX.sym('p', params["np"])
    
    noise_scale = params['measure_std_base']
    noise_scale *= 1 + 10 * huberLoss(x[1], sigma=1e-2)
    y = x + noise_scale @  v
    
    return ca.Function('output_func', [x, v, p], [y])


def get_constr_contr(params):
    """
    create control constraint function of the form h(u, su, p) <= 0
    where u is the control, su is a possible slack variable entering linearly, p is a parameter
    """
    u = ca.SX.sym("u", params["nu"])
    su = ca.SX.sym("su", params["nsu"])
    p = ca.SX.sym("p", params["np"])

    constr = []
    constr += [  u - params["umax"]]
    constr += [ -u + params["umin"] ]
    return ca.Function('h_u', [u, su, p], [ ca.vertcat(*constr)])


def get_constr_state(params):
    """
    create state constraint function of the form h(x, sx, p) <= 0
    where x is the state, sx is a possible slack variable entering linearly, p is a parameter
    """
    x = ca.SX.sym("x", params["nx"])
    sx = ca.SX.sym("sx", params["nsx"])
    p = ca.SX.sym("p", params["np"])

    constr = [ ]
    constr += [  params['rx_min'] - x[0]   ]
    return ca.Function('h_x', [x, sx, p], [ ca.vertcat(*constr)  ])


def get_params():
    """
    Create dictionary of all the parameters
    """

    params = dict()

    # problem dimensions
    params["nx"] = 3                            # state dim
    params["nu"] = 2                            # control dim
    params["ny"] = 3                            # output dim
    params["nw"] = params["nx"]                 # process noise dim
    params["nv"] = params["ny"]                 # output noise dim
    params["np"] = 0                            # number of parameters per (per stage)
    params["nsu"] = 0                           # number of slack vars (contr)
    params["nsx"] = 0                           # number of slack vars (state)
    params["nsxN"] = 0                          # number of slack vars (terminal state)

    # horizon    
    params["N"] = 10        # discrete time horizon
    params["T"] = 3         # continuous time horizon
    
    # slack variable bounds
    params["lbs"] = -np.inf                     # lower bound slack vars
    params["ubs"] = np.inf                      # upper bound slack vars

    # control bounds
    params["umin"] = np.array([ -3, - 1/2 * np.pi ])
    params["umax"] = np.array([  3,  1/2 * np.pi ])

    # state bounds
    params['rx_min'] = 0                       # rx >= rx_min

    # initial state and estimation uncertainty of system
    params["x0bar"]   = np.array( [4, 2,  np.pi] )
    params["P0bar"] = np.diag( [.1**2, .1**2, (np.pi/100)**2 ] )

    # constraint violation penalty weight
    params["constr_u_pen"]  = 1e3
    params["constr_x_pen"]  = 1e3
    
    # regularization
    params["epsilon_K"]     = 1e-4
    params["epsilon_beta"]  = 1e-4
    params["epsilon_var"]   = 1e-4

    # noise standard deviations
    params['process_std_scale'] = np.diag( [1e-1, 1e-1, np.pi / 100])
    params['measure_std_base'] = np.diag( [1e-2, 1e-2, np.pi / 100])

    return params