import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

# use seaborn for standard plot colors
import seaborn as sns
color_palette = sns.color_palette('colorblind')
colors = color_palette.copy()
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) 

golden_ratio = (np.sqrt(5)-1.0)/2.0 

def plotTrajectoryInTime(X, U, params, P=None, Phat=None,  Pbrev=None, K_fb=None, gamma=3, X_ref=None, title=None):
    
    N = U.shape[1]
    
    plt.figure(figsize=(5,8))
    if title is not None: plt.title(title)
    
    # positions
    plt.subplot(5,1,1)
    plt.plot([0, N], 2 * [params['rx_min']], 'k--')     # state constraint
    plt.plot(X[0,:])
    if X_ref is not None:
        plt.plot(X_ref[0,:])
    if P is not None:
        tube_lower = [ X[0,k] - gamma * np.sqrt(P[k][0,0]) for k in range(N+1) ]
        tube_upper = [ X[0,k] + gamma * np.sqrt(P[k][0,0]) for k in range(N+1) ]
        plt.fill_between(list(range(N+1)), tube_lower, tube_upper, alpha=.3)
    plt.ylabel(r'position $r^\mathrm{x}$')
    plt.xticks(range(0,N+1,2))
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.xlim((0, N))
    plt.ylim((-.5, 4.5))

    plt.subplot(5,1,2)
    plt.plot(X[1,:])
    if X_ref is not None:
        plt.plot(X_ref[1,:])
    if P is not None and P[0].size > 1:
        tube_lower = [ X[1,k] - gamma * np.sqrt(P[k][1,1]) for k in range(N+1) ]
        tube_upper = [ X[1,k] + gamma * np.sqrt(P[k][1,1]) for k in range(N+1) ]
        plt.fill_between(list(range(N+1)), tube_lower, tube_upper, alpha=.3)
    plt.ylabel(r'position $r^\mathrm{y}$')
    plt.xticks(range(0,N+1,2))
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.xlim((0, N))
    plt.ylim((-1, 3))

    plt.subplot(5,1,3)
    plt.plot(X[2,:])
    if X_ref is not None:
        plt.plot(X_ref[2,:])
    if P is not None and P[0].size > 2:
        tube_lower = [ X[2,k] - gamma * np.sqrt(P[k][2,2]) for k in range(N+1) ]
        tube_upper = [ X[2,k] + gamma * np.sqrt(P[k][2,2]) for k in range(N+1) ]
        plt.fill_between(list(range(N+1)), tube_lower, tube_upper, alpha=.3)
    plt.ylabel(r'orientation $\theta$')
    plt.xticks(range(0,N+1,2))
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.xlim((0, N))
    plt.ylim((0, 2*np.pi))

    t_disc = list(range(X.shape[1]))

    if K_fb is not None:
        if Pbrev is None:
            P_u = [ K @ p @ K.T for K, p in zip(K_fb, P)  ] 
        else:
            P_u = [ K @ ( p + pbrev + pbrev.T + phat ) @ K.T for K, p, phat, pbrev in zip(K_fb, P, Phat, Pbrev)  ] 

    # controls / velocity
    plt.subplot(5,1,4)

    plt.plot([0, N], 2 * [params['umin'][0]], 'k--')
    plt.plot([0, N], 2 * [params['umax'][0]], 'k--')

    plt.step(t_disc, np.concatenate((U[0, :], [np.nan])), where='post')
    if K_fb is not None:
        u0_tube_lower = [ U[0,k] - gamma * np.sqrt(P_u[k][0,0]) for k in range(N) ] 
        u0_tube_upper = [ U[0,k] + gamma * np.sqrt(P_u[k][0,0]) for k in range(N) ] 
        plt.fill_between(t_disc[1:], u0_tube_lower, u0_tube_upper, alpha=.3, step='pre')
    plt.xlabel(r'discrete time $k$')
    plt.ylabel(r'control $v$')
    plt.xticks(range(0,N+1,2))
    plt.xlim((0, N))
    plt.ylim((1.2 * params["umin"][0], 1.2 * params["umax"][0] ))

    plt.subplot(5,1,5)
    plt.plot([0, N], 2 * [params['umin'][1]], 'k--')
    plt.plot([0, N], 2 * [params['umax'][1]], 'k--')

    plt.step(t_disc, np.concatenate((U[1, :], [np.nan])), where='post')
    if K_fb is not None and P[0].size > 1:
        u1_tube_lower = [ U[1,k] - gamma * np.sqrt(P_u[k][1,1]) for k in range(N) ] 
        u1_tube_upper = [ U[1,k] + gamma * np.sqrt(P_u[k][1,1]) for k in range(N) ] 
        plt.fill_between(t_disc[1:], u1_tube_lower, u1_tube_upper, alpha=.3, step='pre')
    plt.xlabel(r'discrete time $k$')
    plt.ylabel(r'control $\omega$')
    plt.xticks(range(0,N+1,2))
    plt.xlim((0, N))
    plt.ylim( (1.2 * params["umin"][1], 1.2 * params["umax"][1] ))

    plt.tight_layout()


def plotTrajectoryInSpace(X, params, P=None, gamma=3, title=None, newfig=True):

    if newfig: plt.figure()
    if title is not None: plt.title(title)
    if P is not None:
        plotUncertaintyEllipsoids(X, P, color=colors[0], gamma=gamma)
    plotRobotWithOrientation(X, color=colors[0])
    plotConstraintState(params)

    plt.axis('equal')
    plt.xlabel(r'position $r^\mathrm{x}$')
    plt.ylabel(r'position $r^\mathrm{y}$')
    plt.ylim((-.5, 2.5))
    plt.xlim((-.05, 4.1))


def plotCompareTrueAndEstimatedTraj(Xtrue, Xhat, params, Phat=None, gamma=3, title=None, newfig=True, legend=True):
    plotTrajectoryInSpace(Xhat, params, P=Phat, gamma=gamma, title=title, newfig=newfig)
    plotRobotWithOrientation(Xtrue, color=colors[3] )


def plotCompareMPCTrajInSpace(traj, params, gamma=3, title=None):
    X_true_nom , U_true_nom , X_hat_nom , P_hat_nom , _ = traj[0]
    X_true_ol  , U_true_ol  , X_hat_ol  , P_hat_ol  , _ = traj[1]
    X_true_cl  , U_true_cl  , X_hat_cl  , P_hat_cl  , _ = traj[2]
    X_true_dual, U_true_dual, X_hat_dual, P_hat_dual, _ = traj[3]

    plt.figure(figsize=(3.5, 1 * 3.5 * golden_ratio))
    plt.subplot(221)
    plotCompareTrueAndEstimatedTraj(X_true_nom, X_hat_nom, params, Phat=P_hat_nom, gamma=gamma, newfig=False, legend=False)
    plt.xlabel(None)
    plt.xticks([])
    plt.title('Nominal MPC')

    plt.subplot(222)
    plotCompareTrueAndEstimatedTraj(X_true_ol, X_hat_ol, params, Phat=P_hat_ol, gamma=gamma, newfig=False, legend=False)
    plt.title('Open-loop SMPC')
    plt.xlabel(None)
    plt.ylabel(None)
    plt.yticks([])
    plt.xticks([])

    plt.subplot(223)
    plotCompareTrueAndEstimatedTraj(X_true_cl, X_hat_cl, params, Phat=P_hat_cl, gamma=gamma, newfig=False, legend=False)
    plt.title('State-feedback SMPC')

    plt.subplot(224)
    plotCompareTrueAndEstimatedTraj(X_true_dual, X_hat_dual, params, Phat=P_hat_dual, gamma=gamma, newfig=False, legend=False)
    plt.title('Output-feedback SMPC')
    plt.ylabel(None)
    # plt.gca().axes.yaxis.set_ticklabels([])
    plt.yticks([])

def plotCompareConstraintViolation(traj_list, params):

    plt.figure(figsize=[5, 5 * (np.sqrt(5)-1.0)/2.0 ])
    N = max( [X.shape[1] for X, _ in traj_list] ) - 1
    # constraint
    plt.plot([0, N], 2 * [params['rx_min']], 'k--')

    # trajectories
    for X, label in traj_list:
        plt.plot(X[0,:], label=label)

    plt.legend()
    plt.xlabel(r'discrete time $k$')
    plt.ylabel(r'position $r^x$')
    plt.xticks(range(0,N+1,2))
    plt.xlim((0, N))
    plt.ylim((-.5, 1))


def plotCompareConstraintViolationMany(trajs, params, N):

    plt.figure(figsize=(3.5, 0.9 * 3.5 * golden_ratio))
    plt.subplot(221)
    plotSamplesConstraintViol(trajs, 0, N, params)
    plt.ylabel(r'position $r^\mathrm{x}$')
    plt.title("Nominal MPC")
    plt.xticks([])

    plt.subplot(222)
    plotSamplesConstraintViol(trajs, 1, N, params)
    plt.title("Open-loop SMPC")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(223)
    plotSamplesConstraintViol(trajs, 2, N, params)
    plt.xlabel(r'discrete time $k$')
    plt.ylabel(r'position $r^\mathrm{x}$')
    plt.title("State-feedback SMPC")

    plt.subplot(224)
    plotSamplesConstraintViol(trajs, 3, N, params)
    plt.xlabel(r'discrete time $k$')
    plt.title("Output-feedback SMPC")
    plt.yticks([])


def plotSamplesConstraintViol(trajs, idx, N, params):

    for tr in trajs[1:]:
        X = tr[idx][0]
        plt.plot(X[0,:], color=colors[0], alpha=.5, lw=.8)
    plt.plot([0, N], 2 * [params['rx_min']], 'k--', lw=1)
    plt.xticks(range(0,N+1,5))
    plt.xlim((0, N))
    plt.ylim((-.35, 1))


def plotCompareMPCtrajInSpaceAndConstraintViolationMany(trajs, params, N, gamma=3, traj_idx=1):

    traj = trajs[traj_idx]
    X_true_nom , U_true_nom , X_hat_nom , P_hat_nom , _ = traj[0]
    X_true_ol  , U_true_ol  , X_hat_ol  , P_hat_ol  , _ = traj[1]
    # X_true_cl  , U_true_cl  , X_hat_cl  , P_hat_cl  , _ = traj[2]
    X_true_dual, U_true_dual, X_hat_dual, P_hat_dual, _ = traj[3]

    plt.figure(figsize=(3.5, 2.5 * .5 * 3.5 * golden_ratio))

    plt.subplot(321)
    plotCompareTrueAndEstimatedTraj(X_true_nom, X_hat_nom, params, Phat=P_hat_nom, gamma=gamma, newfig=False, legend=False)
    plt.xlabel(None)
    # plt.gca().axes.xaxis.set_ticklabels([])
    plt.xticks([])
    plt.title('Nominal MPC')

    plt.subplot(323)
    plotCompareTrueAndEstimatedTraj(X_true_ol, X_hat_ol, params, Phat=P_hat_ol, gamma=gamma, newfig=False, legend=False)
    plt.title('Open-loop SMPC')
    plt.xlabel(None)
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.xticks([])

    plt.subplot(325)

    plotCompareTrueAndEstimatedTraj(X_true_dual, X_hat_dual, params, Phat=P_hat_dual, gamma=gamma, newfig=False, legend=False)
    plt.title('Output-feedback SMPC')

    plt.subplot(322)
    plotSamplesConstraintViol(trajs, 0, N, params)
    plt.ylabel(r'position $r^\mathrm{x}$')
    plt.title("Nominal MPC")
    plt.xticks([])

    plt.subplot(324)
    plotSamplesConstraintViol(trajs, 1, N, params)
    plt.ylabel(r'position $r^\mathrm{x}$')
    plt.title("Open-loop SMPC")
    plt.xticks([])

    plt.subplot(326)
    plotSamplesConstraintViol(trajs, 3, N, params)
    plt.ylabel(r'position $r^\mathrm{x}$')
    plt.xlabel(r'discrete time $k$')
    plt.title("Output-feedback SMPC")


def plotMPCSuccessRatio(traj_list, params, constr_state):

    plt.figure(figsize=(3.5, 3.5 * .7 * golden_ratio))
    bars = [[] for _ in range(2)] 

    controllers = ['Nominal MPC', 'OLSMPC', 'SFSMPC', 'OFSMPC']
    for idx in range(len(controllers)):
        ratios = computeMPCsuccessRatio(traj_list, idx, params, constr_state)
        for b, r in zip(bars, ratios):
            b += [r[0]]
            print('{:14s}, {:19s}: {}'.format(controllers[idx], r[1], r[0]))

    bars = np.array(bars)
    legend = [ r[1] for r in ratios]
    
    plt.bar( controllers, bars[0,:], width=.25, label=legend[0])
    plt.bar( controllers, bars[1,:], width=.25, bottom = np.sum(bars[:1,:], axis=0), label=legend[1])
    plt.xlim(-.5, 2.5)
    plt.legend(legend, bbox_to_anchor=(.5, 1.02), loc='lower center', ncol=len(legend), columnspacing=0.7, handletextpad=0.4 )
    plt.ylabel('ratio')


def computeMPCsuccessRatio(traj_list, idx, params, constr_state):

    N_traj = len(traj_list) - 1
    ratios = []
    ratios += [[0, 'success']]
    # ratios += [[0, 'constraint violated']]
    ratios += [[0, 'solver failed']]

    # count all the cases
    for i, traj in enumerate(traj_list[1:]):
        X_true, U_true , X_hat , P_hat , return_status  = traj[idx]

        # if constraintViolated( X_true, constr_state) or return_status == 'constr_state_violated':
        #     ratios[1][0] += 1
        if return_status == 'solver_failed':
            ratios[1][0] += 1
        elif return_status == 'success' or return_status is None:
            ratios[0][0] += 1
        else:
            # the above cases should be exhaustive
            print("This should not be printed.")
    
    # compute ratio
    for ra in ratios:
        ra[0] /= N_traj

    return ratios


def plotConstraintState(params):
    plt.plot( 2 * [params['rx_min']]  , [-2, 4], 'k--')


def plotUncertaintyEllipsoids(X, P, color='b', alpha=.3, gamma=3):

    P_surf = [ X[:2,k][:,None] + ellipsoid_surface_2D( gamma**2 * P[k][:2,:2]) for k in range(len(P)) ]
    for p in P_surf:
        plt.fill(p[0, :], p[1,:], color=color, alpha=alpha, edgecolor=None)


def plotUncertaintyIntervalInXdir(X, P, color='b', alpha=.3, gamma=3):

    rx_std = np.sqrt(P).squeeze()

    rx_upper = X[0,:] + gamma * rx_std
    rx_lower = X[0,:] - gamma * rx_std

    for k in range(X.shape[1]):
        plt.plot([rx_lower[k], rx_upper[k]], 2 * [X[1,k]], '-|',color=color, alpha=alpha)
    plt.plot(X[0,:], X[1,:], ':x',  color=color) 


def ellipsoid_surface_2D(P, n=100):
    lam, V = np.linalg.eig(P)
    phi = np.linspace(0, 2 * np.pi, n)
    a = (V @ np.diag(np.sqrt(lam))) @ np.vstack([np.cos(phi), np.sin(phi)])
    return a


def plotRobotWithOrientation(X, color=None):

    if color is None:
        color = 'k'
    plt.plot( X[0,:] , X[1,:], '-',linewidth=.5, color='k')

    for k in range(X.shape[1]):
        marker = generate_marker_rotated_triangle( X[2,k] )
        markersize = 50
        plt.scatter( X[0,k] , X[1,k], color=color, marker=marker, s=markersize, edgecolor='black', linewidths=.3, zorder=10)


def generate_marker_rotated_triangle(angle):

    # non-rotated triangle pointing to right
    marker = np.array([[-.8, .5], [-.8, -.5], [1, 0], [-.8, .5]])
    rotation_mat = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    marker = marker @ rotation_mat
    marker = mpl.path.Path(marker)
    return marker
