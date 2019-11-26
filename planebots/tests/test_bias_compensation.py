# from vision import coverage_control
import logging

import numpy as np

# logger = logging.getLogger(__name__)
# import unittest
# import matplotlib.pyplot as plt
# from scipy.spatial import Voronoi, voronoi_plot_2d
# import skfmm
logger = logging.getLogger("vision")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
# import vision
# import datetime
# from scipy import interpolate
# from vision import controllers
import filterpy
from filterpy.kalman import MerweScaledSigmaPoints
# dt =0.1
# sigmas = MerweScaledSigmaPoints(5, alpha=.1, beta=2., kappa=1.)

# import vision.model as model
from matplotlib import pyplot as plt


def normalize_angle(x):
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    if x > np.pi:  # move to [-pi, pi)
        x -= 2 * np.pi
    return x


# def residual_h_vis(a, b):
#     y = a - b
#     y[2] = normalize_angle(y[2])
#     return y

def residual_x(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    if len(a) > 5:
        y[7] = normalize_angle(y[7])
    return y


def residual_z(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    if len(a) > 3:
        y[5] = normalize_angle(y[5])
    return y


def utov(lr):
    """Converts elisa byte inputs values to velocities and rotation in m/s and rad/s"""
    l, r = lr
    b = 0.0408
    k = 200  # to conver to m/s 120 ^= 0.6m/s
    u_vau, u_omega = np.array([0.5 * (l + r), (r - l) / b]) / k  # wheelbase of 40.8mm
    return [u_vau, u_omega]


def f_kalman(X, dt=1, u=None, linear=False):
    """
    x
    ^
    |  theta <-
    |           \
    |            |
     - - - - - - - ->y
    :param x: state
    :param u: reference [v,theta_dot]
    :return: x new state
    """
    X = list(np.array(X))  # x dx ddx y dy ddy th dth
    Xk1 = np.array([0., 0., 0., 0., 0.])  # x y th v dt
    # x, dx, ddx, y, dy, ddy, th, omega = X
    logger.debug("")
    # rm = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]] ).squeeze()
    b = 0.0408
    k = 200  # to conver to m/s 120 ^= 0.6m/s
    if type(u) is not type(None):
        l, r = u
        u_vau, u_omega = np.array([0.5 * (l + r), (r - l) / b]) / k  # wheelbase of 40.8mm

        if abs(X[4]) < 0.001 or linear:  # if very low angular velocity, approach as a straight line
            dxk = np.cos(X[2]) * X[3] * dt
            dyk = -np.sin(X[2]) * X[3] * dt
            dthk = X[4] * dt
        else:
            R = X[3] / X[4]
            dxk = R * (np.sin(X[2] + X[4] * dt) - np.sin(X[2]))
            dyk = R * (np.cos(X[2] + X[4] * dt) - np.cos(X[2]))
            dthk = X[4] * dt

        Xk1[0] = X[0] + dxk
        Xk1[1] = X[1] + dyk
        Xk1[2] = (X[2] + dthk)
        Xk1[3] = u_vau
        Xk1[4] = u_omega
        if Xk1[1] > 0.3:
            logger.debug("")
    else:
        if abs(X[4]) < 0.001 or linear:  # if very low angular velocity, approach as a straight line
            dxk = np.cos(X[2]) * X[3] * dt
            dyk = np.sin(X[2]) * X[3] * dt
            dthk = X[4] * dt
        else:
            R = X[3] / X[4]
            dxk = R * (np.sin(X[2] + X[4] * dt) - np.sin(X[2]))
            dyk = R * (np.cos(X[2] + X[4] * dt) - np.cos(X[2]))
            dthk = X[4] * dt
        Xk1[0] = X[0] + dxk
        Xk1[1] = X[1] + dyk
        Xk1[2] = (X[2] + dthk)
        Xk1[3] = X[3]
        Xk1[4] = X[4]
    xor = X
    rv = np.array(Xk1).ravel()
    return rv


def getbias(z1, z2):
    # Retrieve the angle bias from the difference in two angles
    a = z2[2] - z1[2]
    rm = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    bias = z1 - np.matmul(rm, z2)
    return bias


def h_odom(X):
    a = X[7]
    a = X[7]
    rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    Xpos = X[:3]
    Xbias = X[5:8]
    XposT = np.matmul(rm.T, Xpos[:2])
    XbiasT = np.matmul(rm.T, Xbias[:2])
    z1, z2 = XposT - XbiasT
    zout = [z1, z2, X[7] - X[2]]
    return list(zout)


def h_vis(x):
    zret = list(x[:3])
    return zret


def h_comb(X):
    z1 = h_vis(X)
    z2 = h_odom(X)
    zc = np.array([*z1, *z2]).ravel()
    zret = list(zc)
    return zret


def f_bias(X, dt, noise=np.zeros(11), u=None):
    # x_coords = f_kalman(x[:5],dt,u=None,linear=True)
    # x[:5] = x_coords
    X = np.array(X)
    Xnew = np.array(X.copy())
    Xnew[0] = X[0] + np.cos(X[2]) * X[3] * dt
    Xnew[1] = X[1] - np.sin(X[2]) * X[3] * dt
    Xnew[2] = X[2] + X[4] * dt
    if u is not None:
        vels = utov(u)
        Xnew[3], Xnew[4] = vels
    bvel = X[8:11]
    # bvel = bvel*0
    # Xnew[5] = X[5] + dt *bvel[0]
    # Xnew[6] = X[6] + dt *bvel[1]
    # Xnew[7] = X[7] + dt *bvel[2]
    return Xnew + noise


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


def f_bias_old(X, dt):
    x_coords = f_kalman(X[:5], dt, u=None)
    X[:5] = x_coords

    bvel = X[8:]
    X[5:8] = X[5:8] + dt * bvel
    return X


def f_vision(x, dt, noise=np.zeros(5), u=None):
    # x_coords = f_kalman(x[:5],dt,u=None,linear=True)
    # x[:5] = x_coords
    X = x.copy()
    X[0] = X[0] + np.cos(X[2]) * X[3] * dt
    X[1] = X[1] - np.sin(X[2]) * X[3] * dt
    X[2] = X[2] + X[4] * dt
    if u is not None:
        vels = utov(u)
        X[3], X[4] = vels
    return X + noise


def lrtestsignal(l=10., r=11., ln=100):
    v1 = np.zeros(ln) + r
    v2 = np.zeros(ln) + l
    vleft = v1.copy()
    vright = v2.copy()
    vleft[len(v1) // 3:len(v1) * 2 // 3] = v1[len(v1) // 3:len(v1) * 2 // 3]
    vright[len(v1) // 3:len(v1) * 2 // 3] = v1[len(v2) // 3:len(v2) * 2 // 3]
    vleft[len(v1) * 2 // 3:] = v2[len(v1) * 2 // 3:]
    vright[len(v1) * 2 // 3:] = v1[len(v1) * 2 // 3:]
    return vleft, vright


if __name__ == '__main__':
    logging.getLogger("vision").setLevel(logging.DEBUG)
    logger.debug("load files")
    logger.debug("Crafting lissajous input velocities")
    n = 300
    n4 = n // 4
    dt = 0.1
    ts = np.linspace(0, n * dt - dt, n)
    # Filter to test the vision performance and to eliminate bug in the prediction.
    variance_vision = np.array([0.053, 0.050, 0.005]) ** 2 * 10  # Variance = sigma**2 gathered from testdata
    # Assume continuous-time white noise:
    PnCov = np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    Q1_jos = np.diag([0.1, 0.1, 0.1, 0.1, 0.1])
    R_jos = np.diag([0.004, 0.004, 0.23]) * 100

    # Generate the vision test data:
    vl, vr = vleft, vright = lrtestsignal(10., 11., n)

    q_bias = np.array([0.01, 0.01, 0.01])
    q_bias_dt = np.array([0.01, 0.01, 0.01])

    sigmas = MerweScaledSigmaPoints(11, alpha=.1, beta=2., kappa=0)
    ukf = filterpy.kalman.UnscentedKalmanFilter(dim_x=11,
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

    ukf.P = np.eye(11) * [0.01, 0.01, 0.01, 0.1, 0.1, *q_bias, *q_bias_dt] * 10
    Q = np.diag([0.1, 0.1, 0.001, 0.5, 0.5, *q_bias, *q_bias_dt]) * 0.1

    logger.debug("initialize states")
    states = np.zeros((11, n))
    meas_vis = np.zeros((3, n))
    meas_eli = np.zeros((3, n))
    ts = np.linspace(dt, n * dt, n)
    tsv = ts

    # Define True biases: These are the initial starting coordinates and angle of the agent.
    bx = 0.2
    by = 0.2
    bth = np.pi * 3 / 2

    bias = [bx, by, bth]
    x0 = np.zeros(11)
    x0[5:8] = bias  # Define the starting coordinate.
    x0[0:3] = bias  # Define the starting coordinate.
    X = x0
    for i in range(len(ts)):
        X = f_bias(X, dt=dt, u=[vleft[i], vright[i]])
        states[:, i] = X

        idx = i
        amp = np.array([0.01, 0.01, 0.01]) * 1
        noise = np.random.random(3) * amp - amp / 2
        meas_vis[:, idx] = h_vis(X) + noise

        amp = np.array([0.01, 0.01, 0.01])
        noise = np.random.random(3) * amp - amp / 2
        xeli = h_odom(X)
        meas_eli[:, i] = xeli + noise

    logger.debug("Plot the kalman states")
    states_kalm = np.zeros((11, n))
    states_kalm_prior = np.zeros((11, n))
    biases = np.zeros((3, n))
    ukf.x = np.zeros(11)

    for i in range(n):
        if i == 18:
            pass
        if i >= len(tsv) // 2:
            logger.debug("Update filter with only odometry")
            # ukf.x = f_bias(ukf.x,dt*4)
            # continue
        if i < len(tsv) // 2 or True:
            xpp = ukf.x
            ukf.predict(fx=f_bias, dt=dt)
            xpp2 = ukf.x
            civ = ukf.P_prior
            z1 = meas_vis[:, i]
            Xth = np.zeros(11)
            Xth[:3] = z1
            Xth[5:8] = bias

            z2_theoretical = h_odom(Xth)
            z2_practice = h_odom(ukf.x)
            z2 = meas_eli[:, i]
            z = np.array([z1, z2]).ravel()
            bias1 = getbias(z1, z2)
            bias2 = ukf.x[5:8]
            # ukf.x[5:8] = bias1+0.018
            xpp3 = ukf.x
            ukf.update(z, hx=h_comb, R=np.diag([0.01, 0.02, 0.01, 0.1, 0.1, 0.1]) * 0.1)
            # ukf.update(z1,hx=h_vis,R=np.diag([0.02,0.02,0.01])*0.001)
            xpp4 = ukf.x
            xpp4 = ukf.x
            # xukf = ukf.x
            states_kalm[:, i] = ukf.x
            biases[:, i] = bias

    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax.plot(states[0, :], states[1, :], 'k', label='$True$')
    ax.plot(meas_vis[0, :], meas_vis[1, :], '+r', label='Vision')
    ax.plot(meas_eli[0, :], meas_eli[1, :], 'xg', label='Elis')
    ax.plot(states_kalm[0, :], states_kalm[1, :], 'xb', label='Kalman')

    ax2.plot(tsv, states_kalm[0, :], 'xk', label='$Kalman x$')
    ax2.plot(tsv, states_kalm[1, :], 'xr', label='$Kalman y$')
    ax2.plot(tsv, states_kalm[3, :], 'xb', label='$Kalman \\omega$')
    ax2.plot(tsv, 10 * states_kalm[4, :], 'xg', label='$Kalman v$')
    # ax2.plot(tsv,states_kalm[2,:],'xg',label='Kalmantheta')
    ax2.plot(ts, states[0, :], '--k', label='$x$')
    ax2.plot(ts, states[1, :], '--r', label='$y$')
    ax2.plot(ts, states[3, :], '--b', label='$\\omega$')
    ax2.plot(ts, 10 * states[4, :], '--g', label='$v$')

    ax3.plot(tsv, states_kalm[5, :], 'xk', label='biasx')
    ax3.plot(tsv, biases[0, :], '--k', label='biasx')
    ax3.plot(tsv, states_kalm[6, :], 'xb', label='biasy')
    ax3.plot(tsv, biases[1, :], '--b', label='biasy')
    ax3.plot(tsv, states_kalm[7, :], 'xr', label='$bias\\theta$')
    ax3.plot(tsv, list(map(lambda x: normalize_angle(x), biases[2, :])), '--r', label='$bias\\theta$')
    ax.legend()
    ax2.legend()
    ax3.legend()
    plt.show()
