# %%
from datetime import datetime
import numpy as np
import matplotlib.pylab as plt

from ellipsoid_utils import splitSigmaIntoBlocksList
from robot_nl_probdef import get_params, get_dyn_discr, get_dyn_nom, get_output_func, get_cost_stage, get_cost_terminal, get_constr_state, get_constr_contr
from robot_nl_plotutils import plotTrajectoryInTime, plotTrajectoryInSpace, plotCompareTrueAndEstimatedTraj, plotCompareConstraintViolation
from simulate_closed_loop import simulate_closed_loop
from solverNominalMPC import solverNominalMPC
from solverOpenLoopSMPC import solverOpenLoopSMPC
from solverStateFeedbackSMPC import solverStateFeedbackSMPC
from solverOutputFeedbackSMPC import solverOutputFeedbackSMPC
from extendedKalmanFilter import createEKFstepfunc

saveresults = False
showplots = True
run_nom = True
run_ol = True
run_sf = True
run_of = True

outfolder = 'robot_nl_results/'
today_str = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
outfile = outfolder + 'res_' + today_str

# number of experiments (i.e. sampled MPC trajectories per controller)
N_exp = 1
# number of closed loop steps (i.e. closed loop length in discrete time)
N_cl = 20

# problem functions and parameters
params = get_params()
dyn_nom = get_dyn_nom(params)
dyn_discr = get_dyn_discr(params)
meas_func = get_output_func(params)
cost_stage = get_cost_stage(params)
cost_terminal = get_cost_terminal(params)
constr_state = get_constr_state(params)
constr_contr = get_constr_contr(params)

# Estimator in closed loop
EKFstep = createEKFstepfunc(dyn_discr, meas_func)

P0 = params["P0bar"]        # initial estimation error covariance
gamma_conf = 3.0            # confidence level for plotting ellipsoids of normal distr.

# all results will be written into this dictionary
results_dict = {}
results_dict["params"] = params
results_dict["N_cl"] = N_cl
results_dict["traj_closedloop"] = [[]]
results_dict["traj_ocp"] = {}
cl_legend = []
if run_nom:  cl_legend += ['nommpc']
if run_ol:   cl_legend += ['olsmpc']
if run_sf:   cl_legend += ['sfsmpc']
if run_of:   cl_legend += ['ofsmpc']
results_dict["traj_closedloop"][-1] = cl_legend


#%% nominal MPC solver, initial OCP
if run_nom:
    solver_nom = solverNominalMPC(dyn_nom, cost_stage, cost_terminal, params, constr_state=constr_state, constr_contr=constr_contr)
    solver_nom.create_solver()
    # Xinit = np.linspace(params["x0bar"], params["xtarget"], params["N"]+1).T
    Xinit = np.linspace(params["x0bar"], params["x0bar"], params["N"]+1).T
    Uinit = np.zeros((params["nu"], params["N"]))
    solver_nom.set_initial_guess_primal( Xinit, Uinit)
    solver_nom.set_value_x0(params["x0bar"])
    solver_nom.solve()
    Xopt_nom, Uopt_nom = solver_nom.get_sol()
    plotTrajectoryInTime(Xopt_nom, Uopt_nom, params, gamma=gamma_conf, title=r'OCP nominal')
    plotTrajectoryInSpace(Xopt_nom, params, gamma=gamma_conf, title=r'OCP nominal')
    results_dict["traj_ocp"]["nommpc"] = (Xopt_nom, Uopt_nom)
    # initialize SMPC with nominal solution
    Xinit_unc = np.copy(Xopt_nom)
    Uinit_unc = np.copy(Uopt_nom)

#%% open loop stochastic MPC solver, initial OCP
if run_ol:
    solver_ol = solverOpenLoopSMPC(dyn_discr, cost_stage, cost_terminal, params, constr_contr=constr_contr, constr_state=constr_state)
    solver_ol.create_solver()
    solver_ol.set_value_x0(params["x0bar"])
    solver_ol.set_value_P0(P0)
    solver_ol.set_initial_guess_primal( Xinit_unc, Uinit_unc)
    solver_ol.solve()
    Xopt_ol, Uopt_ol, P_opt_ol = solver_ol.get_sol()
    plotTrajectoryInTime(Xopt_ol, Uopt_ol, params, P=P_opt_ol, gamma=gamma_conf, title=r'OCP open loop stochastic')
    plotTrajectoryInSpace(Xopt_ol, params, P=P_opt_ol, gamma=gamma_conf, title=r'OCP open loop stochastic')
    results_dict["traj_ocp"]["olsmpc"] = (Xopt_ol, Uopt_ol, P_opt_ol)

#%% state feedback stochastic MPC solver, initial OCP
if run_sf:
    solver_sf = solverStateFeedbackSMPC(dyn_discr, cost_stage, cost_terminal, params, constr_contr=constr_contr, constr_state=constr_state)
    solver_sf.create_solver()
    solver_sf.set_value_x0(params["x0bar"])
    solver_sf.set_value_P0(P0)
    solver_sf.set_initial_guess_primal( Xinit_unc, Uinit_unc, K_fb=0.1 )
    solver_sf.solve()
    Xopt_sf, Uopt_sf, P_opt_sf, K_fb_opt_sf = solver_sf.get_sol()
    plotTrajectoryInTime(Xopt_sf, Uopt_sf, params, P=P_opt_sf, K_fb=K_fb_opt_sf, gamma=gamma_conf, title=r'OCP state feedback stochastic')
    plotTrajectoryInSpace(Xopt_sf, params, P=P_opt_sf, gamma=gamma_conf,  title=r'OCP state feedback stochastic')
    results_dict["traj_ocp"]["sfsmpc"] = (Xopt_sf, Uopt_sf, P_opt_sf, K_fb_opt_sf)

#%% output feedback stochastic MPC solver, initial OCP
if run_of:
    solver_of = solverOutputFeedbackSMPC(dyn_discr, meas_func, cost_stage, cost_terminal, params, constr_state=constr_state, constr_contr=constr_contr)
    solver_of.create_solver()
    solver_of.set_value_x0(params["x0bar"])
    solver_of.set_value_P0(P0)
    solver_of.set_initial_guess_primal( Xinit_unc, Uinit_unc )
    solver_of.solve()
    Xopt_of, Uopt_of, Sigma_opt_of, K_fb_opt_of, K_ob_opt_of = solver_of.get_sol()
    P_opt_of, Phat_opt_of, Pbrev_opt_of = splitSigmaIntoBlocksList(Sigma_opt_of)
    plotTrajectoryInTime(Xopt_of, Uopt_of, params, P=P_opt_of, Phat=Phat_opt_of, Pbrev=Pbrev_opt_of, K_fb=K_fb_opt_of, gamma=gamma_conf, title=r'OCP dual')
    plotTrajectoryInSpace(Xopt_of, params, P=P_opt_of, gamma=gamma_conf, title=r'OCP output feedback, with $P$ plotted for uncertainty')
    plotTrajectoryInSpace(Xopt_of, params, P=Phat_opt_of, gamma=gamma_conf,  title=r'OCP output feedback, with $\hat P$ plotted for uncertainty')
    results_dict["traj_ocp"]["ofsmpc"] = (Xopt_of, Uopt_of, Sigma_opt_of, K_fb_opt_of, K_ob_opt_of)

if showplots: plt.show()

#%% closed loop
for i in range(N_exp):

    results_dict["traj_closedloop"] += [[]]

    # sample noise (every controller is run with the same noise realization)
    W_samp = np.random.randn(params["nw"], N_cl)        # process noise
    V_samp = np.random.randn(params["nv"], N_cl)        # measurement noise
    w0_samp = np.random.randn(params["nw"])             # initial state noise

    #%% nominal
    if run_nom:
        X_true_nom, U_true_nom, X_hat_nom, P_hat_nom, return_msg = simulate_closed_loop(N_cl, solver_nom, get_dyn_discr(params), meas_func, EKFstep, params,  W=W_samp, V=V_samp,w0=w0_samp)
        results_dict["traj_closedloop"][-1] += [(X_true_nom, U_true_nom, X_hat_nom, P_hat_nom, return_msg)]
        solver_nom.set_initial_guess_primal( Xinit, Uinit)      # re

    #%% open loop stochastic
    if run_ol:
        X_true_ol, U_true_ol, X_hat_ol, P_hat_ol, return_msg = simulate_closed_loop(N_cl, solver_ol, get_dyn_discr(params), meas_func, EKFstep, params,  W=W_samp, V=V_samp,w0=w0_samp)
        results_dict["traj_closedloop"][-1] += [(X_true_ol, U_true_ol, X_hat_ol, P_hat_ol, return_msg)]
        solver_ol.set_initial_guess_primal( Xinit_unc, Uinit_unc )  # reset initial guess for next experiment

    #%% state feedback stochastic
    if run_sf:
        X_true_cl, U_true_cl, X_hat_cl, P_hat_cl, return_msg = simulate_closed_loop(N_cl, solver_sf, get_dyn_discr(params), meas_func, EKFstep, params,  W=W_samp, V=V_samp,w0=w0_samp)
        results_dict["traj_closedloop"][-1] += [(X_true_cl, U_true_cl, X_hat_cl, P_hat_cl, return_msg)]
        solver_sf.set_initial_guess_primal( Xinit_unc, Uinit_unc )     # reset initial guess for next experiment

    #%% output feedback stochastic
    if run_of:
        X_true_of, U_true_of, X_hat_of, P_hat_of, return_msg = simulate_closed_loop(N_cl, solver_of, get_dyn_discr(params), meas_func, EKFstep, params,  W=W_samp, V=V_samp,w0=w0_samp)
        results_dict["traj_closedloop"][-1] += [(X_true_of, U_true_of, X_hat_of, P_hat_of, return_msg)]
        solver_of.set_initial_guess_primal( Xinit_unc, Uinit_unc  )     # reset initial guess for next experiment

# some plots
plotCompareTrueAndEstimatedTraj(X_true_nom, X_hat_nom, params, Phat=P_hat_nom, gamma=gamma_conf, title='Nominal MPC')
plotCompareTrueAndEstimatedTraj(X_true_ol, X_hat_ol, params, Phat=P_hat_ol, gamma=gamma_conf, title='Open loop SMPC')
plotCompareTrueAndEstimatedTraj(X_true_cl, X_hat_cl, params, Phat=P_hat_cl, gamma=gamma_conf, title='State feedback SMPC')
plotCompareTrueAndEstimatedTraj(X_true_of, X_hat_of, params, Phat=P_hat_of, gamma=gamma_conf, title='Ouput feedback SMPC')

# compare constraint viol
traj_list = []
if run_nom : traj_list += [(X_true_nom, 'Nominal MPC')]
if run_ol  : traj_list += [(X_true_ol, 'Open loop SMPC')]
if run_sf  : traj_list += [(X_true_cl, 'State feedback SMPC')]
if run_of: traj_list += [(X_true_of, 'Output feedback SMPC')]
plotCompareConstraintViolation(traj_list, params)
if saveresults: np.save(outfile, results_dict)

if showplots: plt.show()
