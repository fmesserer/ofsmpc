import casadi as ca
import numpy as np

class solverNominalMPC:
    def __init__(self, dyn_discr, cost_stage, cost_terminal, params, constr_state=None, constr_contr=None, constr_state_term=None):

        self.uncertainty_aware = False
        self.params = params
        self.nx = params['nx']
        self.nu = params['nu']
        self.np = params['np']
        self.nsx = params['nsx']
        self.nsu = params['nsu']
        self.nsxN = params['nsxN']
        self.N = params['N']
        self.lbs = params['lbs']
        self.ubs = params['ubs']

        self.dyn_discr = dyn_discr
        self.cost_stage = cost_stage
        self.cost_terminal = cost_terminal
        self.constr_state = constr_state
        self.constr_state_term = constr_state_term
        self.constr_contr = constr_contr
        self.init_guess = 0


    def create_solver(self):

        nx = self.nx
        nu = self.nu
        N = self.N

        # build NLP
        # decision variables
        # trajectory
        X = ca.SX.sym('X', nx, N+1)
        U = ca.SX.sym('U', nu, N)
        traj = ca.veccat(X, U)
        # slack variables (if any)
        Sx = ca.SX.sym('Sx', self.nsx, N+1)
        Su = ca.SX.sym('Su', self.nsu, N)
        SxN = ca.SX.sym('Su', self.nsxN, 1)
        S = ca.veccat(Sx, Su, SxN)
        decvar = ca.veccat(traj, S)

        # box constraints on decision vars
        lb_decvar = np.concatenate((-np.inf * np.ones( traj.shape[0]), self.lbs * np.ones(S.shape[0])))
        ub_decvar = np.concatenate(( np.inf * np.ones( traj.shape[0]), self.ubs * np.ones(S.shape[0])))

        # time dependent parameters
        Par_td = ca.SX.sym('Par_td', self.np, N+1)
        nlp_params = ca.veccat(Par_td)
        self.nlp_params_val = np.zeros(nlp_params.shape)

        # cost
        obj = 0
        for k in range(N):
            obj += self.cost_stage(X[:,k], U[:,k], 0, Sx[:,k], Su[:,k], Par_td[:, k])
        obj += self.cost_terminal(X[:,-1], 0, Sx[:,-1], SxN, Par_td[:,-1])

        # constraints
        g = []
        lbg = []
        ubg = []

        # dynamics
        dyn_map = self.dyn_discr.map(N)
        constr_dyn = dyn_map(X[:,:-1], U, Par_td[:,:-1]) - X[:,1:]
        g.append(ca.vec(constr_dyn))
        lbg.extend([0] * N * nx)
        ubg.extend([0] * N * nx)

        # control constraints
        if self.constr_contr is not None:
            h_u_map = self.constr_contr.map(N)
            constr_u = h_u_map(U, Su, Par_td[:,:-1])
            g.append(ca.reshape(constr_u, -1, 1))
            lbg.extend([-ca.inf] * constr_u.shape[0] * constr_u.shape[1])
            ubg.extend([0]       * constr_u.shape[0] * constr_u.shape[1])

        # state constraints
        if self.constr_state is not None:
            h_x_map = self.constr_state.map(N)
            constr_x = h_x_map(X[:,1:], Sx[:,1:], Par_td[:,1:])
            g.append(ca.reshape(constr_x, -1, 1))
            lbg.extend([-ca.inf] * constr_x.shape[0] * constr_x.shape[1])
            ubg.extend([0]       * constr_x.shape[0] * constr_x.shape[1])

        # terminal state constraints
        if self.constr_state_term is not None:
            constr_xN = self.constr_state_term(X[:,-1], SxN, Par_td[:,1])
            g.append( constr_xN )
            lbg.extend([-ca.inf] * constr_xN.shape[0] )
            ubg.extend([0] * constr_xN.shape[0] )

        nlp = {}
        nlp['x'] = decvar
        nlp['p'] = nlp_params
        nlp['f'] = obj
        nlp['g'] = ca.vertcat(*g)

        opts = {}
        opts['ipopt'] = {}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.solver = solver
        self.decvar = decvar
        self.nlp_params = nlp_params
        self.X = X
        self.U = U
        self.S = S
        self.Sx = Sx
        self.Su = Su
        self.SxN = SxN
        self.Par_td = Par_td
        self.lb_decvar = lb_decvar
        self.ub_decvar = ub_decvar
        self.lbg = lbg
        self.ubg = ubg
        self.defined_init_vals = False


    def set_initial_guess_primal(self, X, U, Sx=None, Su=None, SxN=None):
        
        init_guess_ = [X, U]

        if Sx is None:
            Sx = ca.DM.zeros(self.Sx.shape)
        if Su is None:
            Su = ca.DM.zeros(self.Su.shape)
        if SxN is None:
            SxN = ca.DM.zeros(self.SxN.shape)
        init_guess_ += [Sx, Su, SxN]
        self.init_guess = ca.veccat(*init_guess_)


    def set_value_par(self, Par):
        self.nlp_params_val = ca.veccat(Par)


    def set_value_x0(self, x0_lb, x0_ub=None):
        """
        set bounds on value of x0
        """
        if x0_ub is None:
            x0_ub = x0_lb
        # set x0 by adding to box constr
        self.lb_decvar[:self.nx] = np.array(x0_lb).squeeze()
        self.ub_decvar[:self.nx] = np.array(x0_ub).squeeze()
        self.defined_init_vals = True


    def get_value_u0(self):
        return self.Uopt[:,0]


    def get_sol(self):
        return self.Xopt, self.Uopt


    def solve(self):

        assert(self.defined_init_vals) 

        sol = self.solver( \
            x0  = self.init_guess, \
            p   = self.nlp_params_val, \
            lbx = self.lb_decvar, \
            ubx = self.ub_decvar, \
            ubg = self.ubg, \
            lbg = self.lbg, \
        )

        Xopt = ca.evalf(ca.substitute(self.X, self.decvar, sol['x'])).full()
        Uopt = ca.evalf(ca.substitute(self.U, self.decvar, sol['x'])).full()
        Sx_opt = ca.evalf(ca.substitute(self.Sx, self.decvar, sol['x'])).full()
        Su_opt = ca.evalf(ca.substitute(self.Su, self.decvar, sol['x'])).full()
        SxN_opt = ca.evalf(ca.substitute(self.SxN, self.decvar, sol['x'])).full()

        self.sol = sol
        self.Xopt = Xopt
        self.Uopt = Uopt
        self.set_initial_guess_primal(Xopt, Uopt, Sx=Sx_opt, Su=Su_opt, SxN=SxN_opt)
        return self.solver.stats()['success']


    def set_initial_guess_by_shifting(self):

        Xopt_shift = np.roll(self.Xopt, -1, axis=1)
        Xopt_shift[:, -1] = Xopt_shift[:,-2]
        Uopt_shift = np.roll(self.Uopt, -1, axis=1)
        Uopt_shift[:, -1] = Uopt_shift[:,-2]

        self.set_initial_guess_primal(Xopt_shift, Uopt_shift)
