import casadi as ca
import numpy as np

from ellipsoid_utils import get_backoff_constr_contr_w_observer, get_backoff_constr_state, get_covariance_dynamics_augmented_kalman, symmToVec, vecToSymm
from utils import expectation_over_relu


class solverOutputFeedbackSMPC:
    def __init__(self, dyn_discr, output_func, cost_stage, cost_terminal, params, constr_state=None, constr_contr=None, constr_state_term=None):

        self.epsilon_K = params["epsilon_K"]
        self.epsilon_beta = params["epsilon_beta"]
        self.epsilon_var = params["epsilon_var"]
        self.constr_x_pen = params["constr_x_pen"]
        self.constr_u_pen = params["constr_u_pen"]

        self.uncertainty_aware = True
        self.params = params
        self.nx = params['nx']
        self.nu = params['nu']
        self.ny = params['ny']
        self.np = params['np']
        self.nsx = params['nsx']
        self.nsu = params['nsu']
        self.nsxN = params['nsxN']
        self.N = params['N']
        self.T = params['T']
        self.dt = params["T"] / params["N"]

        self.lbs = params['lbs']
        self.ubs = params['ubs']


        self.dyn_discr = dyn_discr
        self.output_func = output_func
        self.cost_stage = cost_stage
        self.cost_terminal = cost_terminal
        self.dyn_unc = get_covariance_dynamics_augmented_kalman(dyn_discr, output_func, params)
        self.dyn_unc_mapaccum = self.dyn_unc.mapaccum(params["N"])

        self.constr_state = constr_state
        self.constr_contr = constr_contr
        self.constr_state_term = constr_state_term

        self.n_hx = 0
        self.n_hu = 0
        self.n_hxN = 0
        if self.constr_state:
            self.n_hx = constr_state.size1_out(0)
            self.constr_state_bo = get_backoff_constr_state(constr_state, params)
            self.constr_state_bo_map = self.constr_state_bo.map(params["N"])
        if self.constr_contr:
            self.n_hu = constr_contr.size1_out(0)
            self.constr_contr_bo = get_backoff_constr_contr_w_observer(constr_contr, params)
            self.constr_contr_bo_map = self.constr_contr_bo.map(params["N"] - 1)
        if self.constr_state_term:
            self.n_hxN = constr_state.size1_out(0)
            self.constr_state_term_bo = get_backoff_constr_state(constr_state, params, terminal=True)


    def create_solver(self):
            
        nx = self.nx
        nu = self.nu
        N = self.N
        params = self.params

        # build NLP
        # decision variables
        # trajectory
        Xbar = ca.SX.sym('Xbar', nx, N+1)
        Ubar = ca.SX.sym('Ubar', nu, N)
        traj_nom = ca.veccat(Xbar, Ubar)

        # slacks (if any)
        Sx = ca.SX.sym('Sx', self.nsx, N+1)
        Su = ca.SX.sym('Su', self.nsu, N)
        SxN = ca.SX.sym('Su', self.nsxN, 1)
        S = ca.veccat(Sx, Su, SxN)

        self.idx0_Sigma_vec = traj_nom.shape[0] + S.shape[0]  
        Sigma_vec_all = ca.SX.sym('Sigma_vec', self.dyn_unc.size1_in(0), N+1)
        K_fb_vec_dec = ca.SX.sym('K_fb_vec', params["nx"] * params["nu"], params["N"] - 1)
        K_fb_vec_all = ca.horzcat(np.zeros(params["nx"] * params["nu"]), K_fb_vec_dec)
        traj_unc = ca.veccat(Sigma_vec_all, K_fb_vec_dec)
  
        # slacks for constraint variances
        beta_u_all  = ca.SX.sym('beta_u_all' , self.n_hu , N - 1) 
        beta_x_all  = ca.SX.sym('beta_x_all' , self.n_hx , N    ) 
        beta_xN_all = ca.SX.sym('beta_xN_all', self.n_hxN, 1    ) 
        beta = ca.veccat(beta_u_all, beta_x_all, beta_xN_all)
        decvar = ca.veccat(traj_nom, S, traj_unc, beta)

        lb_decvar = np.concatenate((-np.inf * np.ones( traj_nom.shape[0]), self.lbs * np.ones(S.shape[0]), -np.inf * np.ones( traj_unc.shape[0]), np.zeros(beta.shape[0])         ))
        ub_decvar = np.concatenate(( np.inf * np.ones( traj_nom.shape[0]), self.ubs * np.ones(S.shape[0]),  np.inf * np.ones( traj_unc.shape[0]), np.inf * np.ones(beta.shape[0]) ))

        # timedep parameters
        Par_td = ca.SX.sym('Par_td', self.np, N+1)
        self.Par_td_val = ca.DM.zeros(Par_td.shape)
        nlp_params = ca.veccat(Par_td)
        self.pars_val = 0

        # cost
        obj = 0
        for k in range(N):
            # project uncertainty in (x, hat x) to uncertainty in (x, u)
            Sigma_ = vecToSymm( Sigma_vec_all[:,k], 2*nx )
            K_ = ca.reshape( K_fb_vec_all[:,k], nu, nx)
            K_proj = ca.blockcat([[ca.SX_eye(nx), ca.SX.zeros(nx, nx)], [K_, K_]])
            Sigma_xu = K_proj @ Sigma_ @ K_proj.T
            Sigma_xu_vec = symmToVec(Sigma_xu)
            obj += self.cost_stage(Xbar[:,k], Ubar[:,k],  Sigma_xu_vec, Sx[:,k], Su[:,k], Par_td[:, k])

        Sigma_ = vecToSymm( Sigma_vec_all[:,-1], 2*nx )
        Sigma_x = Sigma_[:nx,:nx]
        Sigma_x_vec = symmToVec(Sigma_x)
        obj += self.cost_terminal(Xbar[:,-1], Sigma_x_vec,  Sx[:,-1], SxN, Par_td[:,-1])

        # regularization K_fb
        obj += self.epsilon_K * ca.sumsqr(K_fb_vec_dec)
        obj += self.epsilon_beta * ca.sumsqr(beta)

        # constraints
        g = []
        lbg = []
        ubg = []

        # dynamics
        dyn_map = self.dyn_discr.map(N)
        constr_dyn = dyn_map(Xbar[:,:-1], Ubar, 0, Par_td[:,:-1]) - Xbar[:,1:]
        g.append(ca.vec(constr_dyn))
        lbg.extend([0] * N * nx)
        ubg.extend([0] * N * nx)

        # dynamics uncertainty
        dyn_unc_map = self.dyn_unc.map(N)
        Sigma_plus_vec, K_ob_vec_all = dyn_unc_map(Sigma_vec_all[:,:-1], K_fb_vec_all, Xbar[:,:-1], Ubar, Xbar[:,1:], Par_td[:,:-1] )
        constr_dyn_unc = Sigma_plus_vec - Sigma_vec_all[:,1:]
        g.append(ca.vec(constr_dyn_unc))
        lbg.extend([0] * constr_dyn_unc.shape[0] * constr_dyn_unc.shape[1])
        ubg.extend([0] * constr_dyn_unc.shape[0] * constr_dyn_unc.shape[1])

        # state constraints
        # constraints are enforced via soft constraint, expected value of penalty
        if self.constr_state:
            h_x_map = self.constr_state.map(N)
            constr_x_mean = h_x_map(Xbar[:,1:], Sx[:,1:], Par_td[:,1:])

            # fix slack to variance in constraint dir
            constr_beta_x_var = beta_x_all - self.constr_state_bo_map( Xbar[:,1:], Sigma_vec_all[:,1:], Par_td[:,1:])
            g.append(ca.vec(constr_beta_x_var))
            lbg.extend([0] * self.n_hx * N)
            ubg.extend([np.inf] * self.n_hx * N)
            constr_x_std = ca.sqrt(self.epsilon_var + beta_x_all)
            constr_x_exv = expectation_over_relu(constr_x_mean, constr_x_std)
            obj += ca.sum1(ca.sum2(self.constr_x_pen * constr_x_exv))

        # control constraint
        if self.constr_contr:
            h_u_map = self.constr_contr.map(N)
            constr_u = h_u_map(Ubar, Su, Par_td[:,:-1])

            # nominal constr
            g.append(ca.reshape(constr_u, -1, 1))
            lbg.extend([-ca.inf] * constr_u.shape[0] * constr_u.shape[1])
            ubg.extend([0]       * constr_u.shape[0] * constr_u.shape[1])

            # future constraints are enforced via soft constraint, expected value of penalty
            constr_u_mean = constr_u[:,1:]

            # fix slack to variance in constraint dir
            constr_beta_u_var = beta_u_all - self.constr_contr_bo_map( Ubar[:,1:], Sigma_vec_all[:,1:-1], K_fb_vec_dec, Par_td[:,1:-1])
            g.append(ca.vec(constr_beta_u_var))
            lbg.extend([0] * self.n_hu * (N-1))
            ubg.extend([np.inf] * self.n_hu * (N-1))

            constr_u_std = ca.sqrt(self.epsilon_var + beta_u_all)
            constr_u_exv = expectation_over_relu(constr_u_mean, constr_u_std)
            obj += ca.sum1(ca.sum2(self.constr_u_pen * constr_u_exv))


        # terminal state constraints
        if self.constr_state_term:
            constr_xN_mean = self.constr_state_term(Xbar[:,-1], SxN, Par_td[:,1])

            # fix slack to variance in constraint dir
            constr_beta_xN_var = beta_xN_all -  self.constr_state_term_bo( Xbar[:,-1], Sigma_vec_all[:,-1], Par_td[:,-1])
            g.append(ca.vec(constr_beta_xN_var))
            lbg.extend([0] * self.n_hxN )
            ubg.extend([np.inf] * self.n_hxN )

            constr_xN_std = ca.sqrt(self.epsilon_var + beta_xN_all)
            constr_xN_exv = expectation_over_relu(constr_xN_mean, constr_xN_std)
            obj += ca.sum1(ca.sum2(self.constr_x_pen * constr_xN_exv))


        nlp = {}
        nlp['x'] = decvar
        nlp['p'] = nlp_params
        nlp['f'] = obj
        nlp['g'] = ca.vertcat(*g)
        opts = {}
        opts['ipopt'] = {}
        solver_nom = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.solver = solver_nom
        self.decvar = decvar
        self.Xbar = Xbar
        self.Ubar = Ubar
        self.Sigma_vec_all = Sigma_vec_all
        self.K_fb_vec_dec = K_fb_vec_dec
        self.K_fb_vec_all = K_fb_vec_all
        self.K_ob_vec_all = K_ob_vec_all
        self.Sx = Sx
        self.Su = Su
        self.SxN = SxN
        self.beta = beta
        self.Par_td = Par_td
        self.nlp_params = nlp_params
        self.lb_decvar = lb_decvar
        self.ub_decvar = ub_decvar
        self.lbg = lbg
        self.ubg = np.array(ubg).astype(float)
        self.defined_init_vals = False


    def set_initial_guess_primal(self, X, U, Sx=None, Su=None, SxN=None, Sigma=None, K_fb=None, beta=None):

        init_guess_ = [X, U]

        if Sx is None:
            Sx = ca.DM.zeros(self.Sx.shape)
        if Su is None:
            Su = ca.DM.zeros(self.Su.shape)
        if SxN is None:
            SxN = ca.DM.zeros(self.SxN.shape)
        init_guess_ += [Sx, Su, SxN]

        if K_fb is None:
            K_fb = .1
        if isinstance(K_fb, (int, float)):
            K_fb = K_fb * np.ones(self.K_fb_vec_dec.shape)
        if isinstance(K_fb, list):
            K_fb = np.concatenate([K_.T.flatten()[:,None] for K_ in K_fb[1:]], axis=1)
        Sigma_0 = None
        if Sigma is None:
            Sigma_0 = self.Sigma0_val
        if isinstance(Sigma, (int, float)):
            Sigma_0 = Sigma_0 * np.eye(2 * self.nx)
 
        if Sigma_0 is not None:
            # get Sigma_all from forward simulation
            Sigma_0 = ca.evalf(symmToVec(Sigma_0)).full()
            K_fb_all = np.concatenate( (np.zeros((K_fb.shape[0], 1)), K_fb), axis=1)
            Sigma_vec_elim, _ = self.dyn_unc_mapaccum(Sigma_0, K_fb_all, X[:,:-1], U, X[:,1:], self.Par_td_val[:,:-1])
            init_guess_.extend([Sigma_0, Sigma_vec_elim, K_fb] )
        else:
            # assume given initial guess was full trajectory
            init_guess_.extend([Sigma, K_fb] )

        if beta is None:
            beta = 1e-8
        if isinstance(beta, (int, float)):
            beta = beta * np.zeros(self.beta.shape[0])
        init_guess_.append(beta)

        self.init_guess = ca.veccat(*init_guess_)


    def set_value_P0(self, P0):
        
        self.Sigma0_val = np.zeros((2*self.nx, 2*self.nx))
        self.Sigma0_val[:self.nx, :self.nx] = P0
        self.Sigma0_val[:self.nx, self.nx:] = -P0
        self.Sigma0_val[self.nx:, :self.nx] = -P0
        self.Sigma0_val[self.nx:, self.nx:] = P0
        Sigma0_val_vec = ca.evalf(symmToVec(self.Sigma0_val)).full().squeeze()

        self.lb_decvar[self.idx0_Sigma_vec:self.idx0_Sigma_vec+Sigma0_val_vec.shape[0]] = Sigma0_val_vec
        self.ub_decvar[self.idx0_Sigma_vec:self.idx0_Sigma_vec+Sigma0_val_vec.shape[0]] = Sigma0_val_vec


    def set_value_par(self, Par_td):

        if isinstance(Par_td, (int, float)):
            Par_td = Par_td * ca.DM.ones(self.Par_td.shape)
        # if Par_td is just a vector
        if len(Par_td.shape) == 1 or Par_td.shape[1] == 1:
            Par_td = ca.repmat(Par_td, 1, self.N+1)
        self.Par_td_val = Par_td
        self.__update_pars_val()


    def __update_pars_val(self):
        '''
        update value of pars_val (e.g. after the value of a component has been set)
        '''
        self.pars_val = ca.veccat(self.Par_td_val)


    def set_value_x0(self, x0_lb, x0_ub=None):
        """
        set bounds on value of x0
        """
        if x0_ub is None:
            x0_ub = x0_lb
       # set x0 by adding to box constr
        self.lb_decvar[:self.nx] = np.array(x0_lb).squeeze()
        self.ub_decvar[:self.nx] = np.array(x0_ub).squeeze()


    def get_value_u0(self):
        return self.Ubar_opt[:,0]


    def get_sol(self):
        return self.Xbar_opt, self.Ubar_opt, self.Sigma_opt, self.K_fb_opt, self.K_ob_opt


    def solve(self):

        sol = self.solver( \
            x0  = self.init_guess, \
            p   = self.pars_val, \
            lbx = self.lb_decvar, \
            ubx = self.ub_decvar, \
            ubg = self.ubg, \
            lbg = self.lbg, \
        )

        X_opt = ca.evalf(ca.substitute(self.Xbar, self.decvar, sol['x'])).full()
        U_opt = ca.evalf(ca.substitute(self.Ubar, self.decvar, sol['x'])).full()
        K_fb_opt_vec = ca.evalf(ca.substitute( self.K_fb_vec_all, self.decvar, sol['x'])).full()
        Sigma_opt_vec = ca.evalf(ca.substitute( self.Sigma_vec_all, self.decvar, sol['x'])).full()
        K_ob_opt_vec = ca.evalf(ca.substitute( self.K_ob_vec_all,   self.decvar, sol['x'])).full()
        Sx_opt = ca.evalf(ca.substitute(self.Sx, self.decvar, sol['x'])).full()
        Su_opt = ca.evalf(ca.substitute(self.Su, self.decvar, sol['x'])).full()
        SxN_opt = ca.evalf(ca.substitute(self.SxN, self.decvar, sol['x'])).full()
        beta_opt = ca.evalf(ca.substitute(self.beta, self.decvar, sol['x'])).full()
        
        self.Xbar_opt = X_opt
        self.Ubar_opt = U_opt
        self.Sigma_opt_vec = Sigma_opt_vec
        self.K_fb_opt_vec = K_fb_opt_vec
        self.K_ob_opt_vec = K_ob_opt_vec

        self.Sigma_opt = [ ca.evalf(vecToSymm(Sigma_opt_vec[:,i], 2*self.params["nx"])).full() for i in range(Sigma_opt_vec.shape[1]) ]
        self.Sigma_opt = [ (Sig + Sig.T) / 2 for Sig in self.Sigma_opt ]   # symmetrize
        self.K_fb_opt = [ ca.reshape(K_fb_opt_vec[:,i], self.nu, self.nx).full() for i in range(K_fb_opt_vec.shape[1]) ]
        self.K_ob_opt = [ ca.reshape(K_ob_opt_vec[:,i], self.nx, self.ny).full() for i in range(K_ob_opt_vec.shape[1]) ]
        self.beta_opt = beta_opt
        self.set_initial_guess_primal(X_opt, U_opt, Sx=Sx_opt, Su=Su_opt, SxN=SxN_opt, Sigma=Sigma_opt_vec, K_fb=K_fb_opt_vec[:,1:], beta=beta_opt)
        return self.solver.stats()['success']


    def set_initial_guess_by_shifting(self):

        Xopt_shift = np.roll(self.Xbar_opt, -1, axis=1)
        Xopt_shift[:, -1] = Xopt_shift[:,-2]
        Uopt_shift = np.roll(self.Ubar_opt, -1, axis=1)
        Uopt_shift[:, -1] = Uopt_shift[:,-2]
        self.set_initial_guess_primal(Xopt_shift, Uopt_shift, K_fb=.1)
