# from vision import coverage_control
import logging

# logger = logging.getLogger(__name__)
# import unittest
# import matplotlib.pyplot as plt
# from scipy.spatial import Voronoi, voronoi_plot_2d
# import skfmm
logger = logging.getLogger("vision")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
from matplotlib import pyplot as plt

from planebots.control.observers import *


# def residual_x(a, b):
#     y = a - b
#     y[2] = normalize_angle(y[2])
#     if len(a) > 5:
#         y[7] = normalize_angle(y[7])
#     return y

# def utov(lr):
#     """Converts elisa byte inputs values to velocities and rotation in m/s and rad/s"""
#     l,r = lr
#     b=0.0408
#     k=200 #to conver to m/s 120 ^= 0.6m/s
#     u_vau,u_omega = np.array([0.5 * (l + r),(r-l) / b])/k# wheelbase of 40.8mm
#     return [u_vau,u_omega]
#

# def getbias(z1,z2):
#     # Retrieve the angle bias from the difference in two angles
#     a = z2[2]-z1[2]
#     rm = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
#     bias = z1-np.matmul(rm,z2)
#     return bias
#
#
# def h_odom(X):
#     a = X[7]
#     a=X[7]
#     rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
#     Xpos = X[:3]
#     Xbias = X[5:8]
#     XposT = np.matmul(rm.T,Xpos[:2])
#     XbiasT = np.matmul(rm.T,Xbias[:2])
#     z1,z2 = XposT-XbiasT
#     zout = [z1,z2,X[7]-X[2]]
#     return list(zout)

# def h_vis(x):
#     zret = list(x[:3])
#     return zret
#
# def h_comb(X):
#     z1 = h_vis(X)
#     z2 = h_odom(X)
#     bias = getbias(z1,z2)
#     zc = np.array([*z1,*z2]).ravel()
#     zret = list(zc)
#     return zret

# def f_bias(X,dt,noise=np.zeros(11),u=None):
#     # x_coords = f_kalman(x[:5],dt,u=None,linear=True)
#     # x[:5] = x_coords
#     X = np.array(X)
#     Xnew = np.zeros_like(X)
#     Xnew[0]   = X[0] + np.cos(X[2])*X[3] * dt
#     Xnew[1]   = X[1] - np.sin(X[2])*X[3] * dt
#     Xnew[2]   = X[2] + X[4] * dt
#
#     if u is not None:
#         vels = utov(u)
#         Xnew[3],Xnew[4] = vels
#     else:
#         Xnew[3] = X[3]
#         Xnew[4] = X[4]
#     bvel = X[8:11]
#     bvel = bvel*0
#     Xnew[5] = X[5] + dt *bvel[0]
#     Xnew[6] = X[6] + dt *bvel[1]
#     Xnew[7] = X[7] + dt *bvel[2]
#     Xnew[8] = X[8]
#     Xnew[9] = X[9]
#     Xnew[10] = X[10]
#     return Xnew + noise

def state_mean(sigmas, Wm):
    x = np.zeros(11)
    sum_sin2, sum_cos2 = 0., 0.
    sum_sin7, sum_cos7 = 0., 0.

    for i in range(len(sigmas)):
        s = sigmas[i]
        x[0] += s[0] * Wm[i]
        x[1] += s[1] * Wm[i]
        x[3] += s[3] * Wm[i]
        x[4] += s[4] * Wm[i]
        x[5] += s[5] * Wm[i]
        x[6] += s[6] * Wm[i]
        x[8] += s[8] * Wm[i]
        x[9] += s[9] * Wm[i]
        x[10] += s[10] * Wm[i]
        # x[11] += s[11] * Wm[i]
        sum_sin2 += np.sin(s[2]) * Wm[i]
        sum_cos2 += np.cos(s[2]) * Wm[i]
        sum_sin7 += np.sin(s[7]) * Wm[i]
        sum_cos7 += np.cos(s[7]) * Wm[i]
    x[2] = np.math.atan2(sum_sin2, sum_cos2)
    x[7] = np.math.atan2(sum_sin7, sum_cos7)
    return x


def lrtestsignal(l=10, r=11, ln=100):
    v1 = np.zeros(ln) + r
    v2 = np.zeros(ln) + l
    vleft = v1.copy()
    vright = v2.copy()
    vleft[len(v1) // 3:len(v1) * 2 // 3] = v1[len(v1) // 3:len(v1) * 2 // 3]
    vright[len(v1) // 3:len(v1) * 2 // 3] = v1[len(v2) // 3:len(v2) * 2 // 3]
    vleft[len(v1) * 2 // 3:] = v2[len(v1) * 2 // 3:]
    vright[len(v1) * 2 // 3:] = v1[len(v1) * 2 // 3:]
    return vleft, vright


from planebots.control import observers

if __name__ == '__main__':
    logging.getLogger("vision").setLevel(logging.DEBUG)
    logger.debug("load files")
    logger.debug("Crafting lissajous input velocities")
    n = 100
    n4 = n // 4
    dt = 1
    ts = np.linspace(0, n * dt - dt, n)
    # Filter to test the vision performance and to eliminate bug in the prediction.
    variance_vision = np.array([0.053, 0.050, 0.005]) ** 2 * 10  # Variance = sigma**2 gathered from testdata
    # Assume continuous-time white noise:
    PnCov = np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    Q1_jos = np.diag([0.1, 0.1, 0.1, 0.1, 0.1]) * 0.1
    R_jos = np.diag([0.004, 0.004, 0.23]) * 100

    q_bias = np.array([0.01, 0.01, 0.01])
    q_bias_dt = np.array([0.01, 0.01, 0.01])
    sigmas = MerweScaledSigmaPoints(11, alpha=.1, beta=2., kappa=0)
    ukf = observers.ukf_bias(dim_x=11,
                             dim_z=6,
                             dt=dt,
                             hx=h_vis,
                             fx=f_bias,
                             points=sigmas,
                             sqrt_fn=None,
                             x_mean_fn=None,
                             z_mean_fn=None,
                             residual_x=residual_x,
                             residual_z=None)

    logger.debug("initialize states")
    states = np.zeros((11, n))
    meas_vis = np.zeros((3, n))
    meas_eli = np.zeros((3, n))
    biases = np.zeros((3, n))
    states_kalm = np.zeros((11, n))
    ts = np.linspace(dt, n * dt, n)
    tsv = ts
    # Define True biases: These are the initial starting coordinates and angle of the agent.
    bx = 0.8
    by = 0.2
    bth = np.pi * 2
    bias = [bx, by, bth]
    x0 = np.zeros(11)
    x0[3] = v = 0.5
    x0[4] = w = 0.1
    x0[5:8] = bias  # Define the starting coordinate.
    x0[0:3] = bias  # Define the starting coordinate.
    ukf.x = x0
    x = x0.copy()
    # Generate the vision test data:
    vl, vr = vleft, vright = lrtestsignal(10., 11., n)
    for i in range(n):
        xnew = f_bias(x, u=[vl[i], vr[i]], dt=dt)
        z1 = h_vis(xnew)
        z2 = h_odom(xnew)

        meas_vis[:, i] = z1
        meas_eli[:, i] = z2
        z = [*z1, *z2]
        x = xnew
        states[:, i] = xnew
    for i in range(n):
        xpp = ukf.x
        ukf.predict(fx=f_bias, dt=dt)
        xpp2 = ukf.x
        civ = ukf.P_prior
        bias_th = getbias(z1, z2)
        bias2 = ukf.x[5:8]
        # ukf.x[5:8] = bias1+0.018
        xpp3 = ukf.x
        zk = [*meas_vis[:, i], *meas_eli[:, i]]
        if not i % 3 or i == 0:
            ukf.update(zk, hx=h_comb, R=np.diag([0.01, 0.02, 0.01, 0.1, 0.1, 0.001]) * 0.1)
        # ukf.update(zk[:3],hx=h_vis,R=np.diag([0.02,0.02,0.01])*0.1)
        xpp4 = ukf.x

        zk = [*meas_vis[:, i], *meas_eli[:, i]]
        states_kalm[:, i] = ukf.x
        # meas_vis[:,i] = z1
        biases[:, i] = bias_th

    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax.plot(states[0, :], states[1, :], 'k', label='$True$')
    ax.plot(meas_vis[0, :], meas_vis[1, :], '+r', label='Vision')
    ax.plot(meas_eli[0, :], meas_eli[1, :], 'xg', label='Elis')
    ax.plot(states_kalm[0, :], states_kalm[1, :], 'xb', label='Kalman')

    ax2.plot(tsv, states_kalm[0, :], 'xk', label='$Kalman x$')
    ax2.plot(tsv, states_kalm[1, :], 'xr', label='$Kalman y$')
    ax2.plot(tsv, states_kalm[2, :], 'xb', label='$Kalman \\omega$')
    ax2.plot(tsv, 10 * states_kalm[3, :], 'xg', label='$Kalman v$')
    ax2.plot(tsv, states_kalm[4, :], 'xc', label='$Kalman w$')
    # ax2.plot(tsv,states_kalm[2,:],'xg',label='Kalmantheta')
    ax2.plot(ts, states[0, :], '--k', label='$x$')
    ax2.plot(ts, states[1, :], '--r', label='$y$')
    ax2.plot(ts, states[2, :], '--b', label='$\\theta$')
    ax2.plot(ts, 10 * states[3, :], '--g', label='$v$')
    ax2.plot(ts, states[4, :], '--c', label='$\\omega$')

    ax3.plot(tsv, states_kalm[5, :], 'xk', label='biasx')
    # ax3.plot(tsv,biases[0,:],'--k',label='biasesx')
    ax3.plot(tsv, states_kalm[6, :], 'xb', label='biasy')
    # ax3.plot(tsv,biases[1,:],'--b',label='biasesy')
    ax3.plot(tsv, states_kalm[7, :], 'xr', label='$bias\\theta$')
    # ax3.plot(tsv,list(map(lambda x: normalize_angle(x),biases[2,:])),'--r',label='$bias\\theta$')
    ax.legend()
    ax2.legend()
    ax3.legend()
    plt.show()
