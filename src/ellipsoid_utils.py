import numpy as np
import casadi as ca

def get_covariance_dynamics(dyn_discr, params):
    """
    in:
        dyn_discr:  dynamics function of the form x_next = dyn_discr(x, u, w, p)
        params:     dictionary with parameters

    out:
        function corresponding to the linearization (at x, u, 0) based covariance dynamics of the form P_next = dyn_cov(P, K, x, u, p),
        where K is a feedback gain and all matrix quantities (P, K, P_next) are represented by vectors
    """

    nx = params["nx"]
    nu = params["nu"]
    nw = params["nw"]
    np = params["np"]
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    w = ca.SX.sym("w", nw)
    p = ca.SX.sym("p", np)

    # compute sensitivities
    x_next = dyn_discr(x, u, w, p)
    A   = ca.jacobian(x_next, x) 
    Gam = ca.jacobian(x_next, w)
    B = ca.jacobian(x_next, u)
    # feedback gain
    Kvec = ca.SX.sym('Kvec', nx * nu)
    K = ca.reshape(Kvec, nu, nx)

    # variable for uncertainty, as vector and symmetric matrix
    Pvec = ca.SX.sym('Pvec', nx*(nx+1) // 2)
    P = vecToSymm(Pvec, nx)
    # dynamics
    P_next =  (A + B @ K) @ P @ (A + B @ K).T + Gam @ Gam.T
    # evaluate at nominal traj
    P_next = ca.substitute(P_next, w, 0)

    # ellipsoid dynamics
    return ca.Function('dyn_cov', [Pvec, Kvec, x, u, p], [ symmToVec(P_next) ])


def get_covariance_dynamics_augmented_kalman(dyn_discr, output_func, params):
    '''
    linearization based covariance dynamics of the estimation error augmented state space.
    A Kalman Filter is used on the linearized dynamics.

    in:
        dyn_discr:      dynamics function of the form x_next = dyn_discr(x, u, w, p)
        output_func:    output function of the form y = output_func(x, v, p)
        params:         dictionary with parameters
    
    out:
        function corresponding to the linearization based covariance dynamics of the form P_next = dyn_cov(P, K, x, x_next, u, p)
        where K is a feedback gain and all matrix quantities (P, K, P_next) are represented by vectors
    '''

    nx = params["nx"]
    nu = params["nu"]
    nw = params["nw"]
    ny = params["ny"]
    nv = params["nv"]
    np = params["np"]
    x = ca.SX.sym("x", nx)
    xp = ca.SX.sym("xp", nx)        # next state as variable
    u = ca.SX.sym("u", nu)
    w = ca.SX.sym("u", nw)
    v = ca.SX.sym("v", nv)
    p = ca.SX.sym("p", np)

    # sensitivities dynamics
    x_next = dyn_discr(x, u, w, p)
    A = ca.substitute(ca.jacobian(x_next, x), w, 0)
    B = ca.substitute(ca.jacobian(x_next, u), w, 0)
    Gam = ca.substitute(ca.jacobian(x_next, w), w, 0)
    # sensitivities output
    # output is measured at next state
    y = output_func(xp, v, p)
    C = ca.substitute(ca.jacobian(y, xp), v, 0)
    D = ca.substitute(ca.jacobian(y, v), v, 0)

    # feedback gain
    K_fb_vec = ca.SX.sym('K_fb_vec', nx * nu)
    K_fb = ca.reshape(K_fb_vec, nu, nx)

    # current covariance
    Sig_vec = ca.SX.sym('Sig_vec', 2*nx*(2*nx+1) // 2)
    Sig = vecToSymm(Sig_vec, 2*nx)
    # define blocks
    Phat = Sig[nx:, nx:]

    # get Kalman gain
    Phat_pred = A @ Phat @ A.T + Gam @ Gam.T
    S = C @ Phat_pred @ C.T + D @ D.T
    # K S = Ppred C.T <=> S.T K.T = C Ppred.T   (and Ppred.T = Ppred) 
    K_ob = ca.solve(S.T, C @ Phat_pred ).T

    # update all in one
    Atilde = ca.blockcat( [[ A + B @ K_fb ,  B @ K_fb ], [ca.SX.zeros( nx, nx ), ( ca.SX_eye(nx) - K_ob @ C ) @ A  ]])
    Gamtilde = ca.blockcat([[ Gam, ca.SX.zeros( nx, nv ) ] , [ (K_ob @ C - ca.SX_eye(nx)) @ Gam, K_ob @ D ]])
    Sig_plus =  Atilde @ Sig @ Atilde.T + Gamtilde @ Gamtilde.T

    return ca.Function('dyn_cov_augmented', [Sig_vec, K_fb_vec, x, u, xp, p], [ symmToVec(Sig_plus), ca.vec(K_ob) ])


def get_backoff_constr_state(h_x, params, terminal=False, augmented_state_uncertainty=True):
    """
    create function that computes the state constraint backoff.
    in:
        h_x:            state constraint function of the form h_x(x, s, p)
        params:         dictionary with parameters
        terminal:       set True for terminal state constraint
        augmented_state_uncertainty:    set True if the state uncertainty is augmented in the state space by the estimation error

    out:
        function corresponding to the state constraint backoff of the form h_x_backoff(x, P, p)
    """

    nx = params["nx"]
    ns = params["nsx"] if not terminal else params["nsxN"]
    np = params["np"]
    x = ca.SX.sym("x", nx)
    s = ca.SX.sym("s", ns)
    p = ca.SX.sym("p", np)
    if augmented_state_uncertainty:
        Pvec = ca.SX.sym('Pvec', 2*nx*(2*nx+1) // 2)
        P = vecToSymm(Pvec, 2*nx)
    else:
        Pvec = ca.SX.sym('Pvec', nx*(nx+1) // 2)
        P = vecToSymm(Pvec, nx)
    # linearize at x
    h_x_jac = ca.jacobian(h_x(x, s, p), x)
    # hack: substitute 0 for s, since h_x_jac should not be function of s anyway
    h_x_jac = ca.substitute(h_x_jac, s, 0)

    nh = h_x_jac.shape[0]
    backoff_ = []
    for i in range( nh):
        # project P onto constraint gradient direction
        backoff_.append( h_x_jac[i,:] @ P[:nx, :nx] @ h_x_jac[i,:].T)

    return  ca.Function('h_x_bo', [x, Pvec, p], [ca.vertcat(*backoff_)])


def get_backoff_constr_contr_w_observer(h_u, params, augmented_state_uncertainty=True):
    """
    create function that computes the control constraint backoff.
    in:
        h_u:            control constraint function of the form h_u(u, s, p)
        params:         dictionary with parameters
        augmented_state_uncertainty:    set True if the state uncertainty is augmented in the state space by the estimation error

    out:
        function corresponding to the control constraint backoff of the form h_u_backoff(u, P, K, p)
    """

    nx = params["nx"]
    nu = params["nu"]
    ns = params["nsu"]
    np = params["np"]
    u = ca.SX.sym("u", nu)
    s = ca.SX.sym("s", ns)
    p = ca.SX.sym("p", np)
    if augmented_state_uncertainty:
        Pvec = ca.SX.sym('Pvec', 2*nx*(2*nx+1) // 2)
        P = vecToSymm(Pvec, 2*nx)
    else:
        Pvec = ca.SX.sym('Pvec', nx*(nx+1) // 2)
        P = vecToSymm(Pvec, nx)

    # feedback gain
    Kvec = ca.SX.sym('Kvec', nx * nu)
    K = ca.reshape(Kvec, nu, nx)

    # state uncertainty P induces control uncertainty
    if augmented_state_uncertainty:
        Kproj = ca.blockcat([[ca.SX_eye(nx), ca.SX.zeros( nx, nx ) ], [ K, K ] ])
        Pproj = Kproj @ P @ Kproj.T
        Pproj = Pproj[nx:,nx:]
    else:
        Pproj = K @ P @ K.T

    # linearize at u
    h_u_jac = ca.jacobian(h_u(u, s, p), u)
    # hack: substitute 0 for s, since h_u_jac should not be function of s anyway
    h_u_jac = ca.substitute(h_u_jac, s, 0)
    
    nh = h_u_jac.shape[0]
    backoff_ = []
    for i in range( nh):
        # project uncertainty onto constraint gradient direction
        backoff_.append( h_u_jac[i,:] @  Pproj @ h_u_jac[i,:].T)

    return ca.Function('h_u_bo', [u, Pvec, Kvec, p], [ca.vertcat(*backoff_)])


def vecToSymm(Pvec, nx):
    """
    in: vector encoding symmetrix matrix (its entries)
    out: corresponding symmetrc matrix
    """
    return ca.tril2symm(ca.SX(ca.Sparsity.lower(nx), Pvec))


def symmToVec(P):
    """
    in: symmetrix matrix
    out: vector encoding of its entries

    assumption P is either np.ndarray or some casadi type
    """
    if isinstance(P, np.ndarray):
        P = ca.DM(P)
    return P[P.sparsity().makeDense()[0].get_lower()]


def vecToLower(Lvec, nx):
    """
    in: vector encoding lower triangular matrix (its entries)
    out: corresponding lower triangular matrix
    """
    return ca.SX(ca.Sparsity.lower(nx), Lvec)


def lowerToVec(L):
    """
    in: lower triangular matrix
    out: vector encoding of its entries

    assumption P is either np.ndarray or some casadi type
    """
    
    if isinstance(L, np.ndarray):
        L = ca.DM(L)
    return L[L.sparsity().makeDense()[0].get_lower()]


def splitSigmaIntoBlocksList(Sigma_list):
    """
    Take a list of Sigmas (corresponding to augmented state uncertainty) and split them into its block components.
    Returns list of the block components.
    """

    P_list = []
    Phat_list = []
    Pbrev_list = []
    for S in Sigma_list:
        P, Phat, Pbrev = splitSigmaIntoBlocks(S)
        P_list.append(P)
        Phat_list.append(Phat)
        Pbrev_list.append(Pbrev)

    return P_list, Phat_list, Pbrev_list


def splitSigmaIntoBlocks(Sigma):
    '''
    input: Sigma = [P, Pbrev.T; Pbrev, Phat ]

    returns P, Phat, Pbrev
    '''

    nx = Sigma.shape[0] // 2
    return Sigma[:nx,:nx], Sigma[nx:,nx:], Sigma[nx:, :nx]
