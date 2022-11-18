import casadi as ca
import numpy as np

from ellipsoid_utils import get_backoff_constr_contr_w_observer, get_backoff_constr_state, get_covariance_dynamics, symmToVec, vecToSymm
from utils import expectation_over_relu

class solverOpenLoopSMPC:
    def __init__(self, dyn_discr, cost_stage, cost_terminal, params, constr_state=None, constr_contr=None, constr_state_term=None):

        self.constr_x_pen = params["constr_x_pen"]
        self.constr_u_pen = params["constr_u_pen"]
        self.epsilon_var = params["epsilon_var"]

        self.uncertainty_aware = True
        self.params = params
        self.nx = params['nx']
        self.nu = params['nu']
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
        self.cost_stage = cost_stage
        self.cost_stage_map = self.cost_stage.map(self.N)
        self.cost_terminal = cost_terminal
        self.dyn_unc = get_covariance_dynamics(dyn_discr, params)
        self.dyn_unc_mapaccum = self.dyn_unc.mapaccum(params["N"])

        self.constr_state = constr_state
        self.constr_contr = constr_contr
        self.constr_state_term = constr_state_term

        if self.constr_state:
            self.n_hx = constr_state.size1_out(0)
            self.constr_state_bo = get_backoff_constr_state(constr_state, params, augmented_state_uncertainty=False)
            self.constr_state_bo_map = self.constr_state_bo.map(params["N"])
        if self.constr_contr:
            self.n_hu = constr_contr.size1_out(0)
            self.constr_contr_bo = get_backoff_constr_contr_w_observer(constr_contr, params, augmented_state_uncertainty=False)
            self.constr_contr_bo_map = self.constr_contr_bo.map(params["N"] - 1)
        if self.constr_state_term:
            self.n_hxN = constr_state.size1_out(0)
            self.constr_state_term_bo = get_backoff_constr_state(constr_state, params, terminal=True, augmented_state_uncertainty=False)


    def create_solver(self):
            
        nx = self.nx
        nu = self.nu
        N = self.N

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
  
        # uncertainty trajectory (Sigma is eliminated)
        Sigma0_vec = ca.SX.sym('Sigma0_vec', self.dyn_unc.size1_in(0))
        K_fb_vec_all = np.zeros((nx*nu, N))
        traj_unc = ca.veccat(Sigma0_vec)
  
        decvar = ca.veccat(traj_nom, S, traj_unc)
        lb_decvar = np.concatenate((-np.inf * np.ones( traj_nom.shape[0]), self.lbs * np.ones(S.shape[0]), -np.inf * np.ones( traj_unc.shape[0]) ))
        ub_decvar = np.concatenate(( np.inf * np.ones( traj_nom.shape[0]), self.ubs * np.ones(S.shape[0]),  np.inf * np.ones( traj_unc.shape[0]) ))

        # timedep parameters
        Par_td = ca.SX.sym('Par_td', self.np, N+1)
        self.Par_td_val = ca.DM.zeros(Par_td.shape)
        # initial uncertainty
        Sigma0bar_vec = ca.SX.sym('Sigma0bar_vec', Sigma0_vec.shape)
        nlp_params = ca.veccat(Sigma0bar_vec, Par_td)
        self.parsval = np.zeros(nlp_params.shape)

        # eliminate uncertainty variance
        Sigma_elim_vec = self.dyn_unc_mapaccum(Sigma0_vec, K_fb_vec_all, Xbar[:,:-1], Ubar, Par_td[:,:-1])
        Sigma_vec_all = ca.horzcat( Sigma0_vec, Sigma_elim_vec )

        # cost
        obj = 0
        for k in range(N):
            # no feedback, so no uncertainty in u
            Sigma_ = vecToSymm( Sigma_vec_all[:,k], nx )
            Sigma_xu = ca.SX(nx+nu, nx+nu)
            Sigma_xu[:nx, :nx] = Sigma_
            Sigma_xu_vec = symmToVec(Sigma_xu)
            obj += self.cost_stage(Xbar[:,k], Ubar[:,k],  Sigma_xu_vec, Sx[:,k], Su[:,k], Par_td[:, k])

        obj += self.cost_terminal(Xbar[:,-1], Sigma_vec_all[:,-1],  Sx[:,-1], SxN, Par_td[:,-1])

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

        # initial value Sigma
        g.append( Sigma0_vec - Sigma0bar_vec)
        lbg.extend( [0] * Sigma0_vec.shape[0] )
        ubg.extend( [0] * Sigma0_vec.shape[0] )

        # state constraints
        # constraints are enforced via soft constraint, expected value of penalty
        if self.constr_state:
            h_x_map = self.constr_state.map(N)
            constr_x_mean = h_x_map(Xbar[:,1:], Sx[:,1:], Par_td[:,1:])
            constr_x_std = ca.sqrt(self.epsilon_var + self.constr_state_bo_map( Xbar[:,1:], Sigma_vec_all[:,1:], Par_td[:,1:]))
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
            constr_u_std = ca.sqrt(self.epsilon_var + self.constr_contr_bo_map( Ubar[:,1:], Sigma_vec_all[:,1:-1], K_fb_vec_all[:,1:], Par_td[:,1:-1]))
            constr_u_exv = expectation_over_relu(constr_u_mean, constr_u_std)
            obj += ca.sum1(ca.sum2(self.constr_u_pen * constr_u_exv))

        # terminal state constraints
        if self.constr_state_term:
            constr_xN_mean = self.constr_state_term(Xbar[:,-1], SxN, Par_td[:,1])
            constr_xN_std = ca.sqrt(self.epsilon_var + self.constr_state_term_bo( Xbar[:,-1], Sigma_vec_all[:,-1]), Par_td[:,1])
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
        self.Sigma0_vec = Sigma0_vec
        self.Sigma_vec_all = Sigma_vec_all
        self.Sx = Sx
        self.Su = Su
        self.SxN = SxN
        self.Par_td = Par_td
        self.Sigma0bar_vec = Sigma0bar_vec
        self.nlp_params = nlp_params
        self.lb_decvar = lb_decvar
        self.ub_decvar = ub_decvar
        self.lbg = lbg
        self.ubg = np.array(ubg).astype(float)
        self.defined_init_vals = False


    def set_initial_guess_primal(self, X, U, Sx=None, Su=None, SxN=None, Sigma_0=None):

        init_guess_ = [X, U]

        if Sx is None:
            Sx = ca.DM.zeros(self.Sx.shape)
        if Su is None:
            Su = ca.DM.zeros(self.Su.shape)
        if SxN is None:
            SxN = ca.DM.zeros(self.SxN.shape)
        init_guess_ += [Sx, Su, SxN]

        if Sigma_0 is None:
            Sigma_0 = self.Sigma0_val
        if isinstance(Sigma_0, (int, float)):
            Sigma_0 = Sigma_0 * np.eye(2 * self.nx)
        Sigma_0 = ca.evalf(symmToVec(Sigma_0)).full()
        init_guess_.append( Sigma_0 )

        self.init_guess = ca.veccat(*init_guess_)


    def set_value_P0(self, P0):
        self.Sigma0_val = np.copy(P0)
        self.__update_pars_val()


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
        self.pars_val = ca.veccat(ca.evalf(symmToVec(self.Sigma0_val)), ca.veccat(self.Par_td_val))


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
        return self.Xbar_opt, self.Ubar_opt, self.Sigma_opt


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
        Sigma_opt_vec = ca.evalf(ca.substitute( self.Sigma_vec_all, ca.veccat(self.decvar, self.nlp_params), ca.veccat(sol['x'], self.pars_val))).full()

        Sx_opt = ca.evalf(ca.substitute(self.Sx, self.decvar, sol['x'])).full()
        Su_opt = ca.evalf(ca.substitute(self.Su, self.decvar, sol['x'])).full()
        SxN_opt = ca.evalf(ca.substitute(self.SxN, self.decvar, sol['x'])).full()
        

        self.Xbar_opt = X_opt
        self.Ubar_opt = U_opt
        self.Sigma_opt_vec = Sigma_opt_vec
        self.Sigma_opt = [ ca.evalf(vecToSymm(Sigma_opt_vec[:,i], self.params["nx"])).full() for i in range(Sigma_opt_vec.shape[1]) ]
        self.Sigma_opt = [ (Sig + Sig.T) / 2 for Sig in self.Sigma_opt ]   # symmetrize

        self.set_initial_guess_primal(X_opt, U_opt, Sx=Sx_opt, Su=Su_opt, SxN=SxN_opt)
        return self.solver.stats()['success']


    def set_initial_guess_by_shifting(self):

        Xopt_shift = np.roll(self.Xbar_opt, -1, axis=1)
        Xopt_shift[:, -1] = Xopt_shift[:,-2]
        Uopt_shift = np.roll(self.Ubar_opt, -1, axis=1)
        Uopt_shift[:, -1] = Uopt_shift[:,-2]

        self.set_initial_guess_primal(Xopt_shift, Uopt_shift)
