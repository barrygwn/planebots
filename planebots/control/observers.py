import logging
import time

import numpy as np

logger = logging.getLogger(__name__)
import planebots
# Implements controllers for controlling an agent
import filterpy
from filterpy.kalman import MerweScaledSigmaPoints
from planebots.output.gctronic import utov
from planebots.control import dwa


# This file houses observers to be used by the experimental platform, as well as functions defining the dynamics of the system.

def remap_angle(x):
    """maps the angle onto the domain [-pi, pi) """
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    if x > np.pi:  # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def residual_x(a, b):
    """Calculate the difference between two vectors defining the state, with a remapping of angles."""
    y = a - b
    y[2] = remap_angle(y[2])
    if len(a) > 5:
        y[7] = remap_angle(y[7])
    return y


def residual_z(a, b):
    """Calculate the difference between two vectors defining the state, with a remapping of angles."""
    y = a - b
    if len(a) >= 3:
        y[2] = remap_angle(y[2])
    if len(a) >= 5:
        y[5] = remap_angle(y[5])

    return y


def getbias(z1, z2):
    """Get the bias x,y,theta with z1 the real position of the agent, and z2 the pose as given by the odometry"""
    a = z2[2] - z1[2]
    rm = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    bias = z1 - np.matmul(rm, z2)
    return bias


def h_odom(X):
    """Measurement function of the odometry, takes care of the mapping from local to global coordinates"""
    a = X[7]
    rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    Xpos = X[:3]
    Xbias = X[5:8]
    XposT = np.matmul(rm.T, Xpos[:2])
    XbiasT = np.matmul(rm.T, Xbias[:2])
    z1, z2 = XposT - XbiasT
    zout = [z1, z2, X[7] - X[2]]
    return list(zout)


def h_uw(X):
    """Measurement function of the odometry, takes care of the mapping from local to global coordinates"""
    zout = [X[3], X[4]]
    return list(zout)


def h_vis(x):
    """First three states are measured by the camera system"""
    zret = list(x[:3])
    return zret


def h_comb(X):
    """Combined measurement using both the camera and the odometry"""
    z1 = h_vis(X)
    z2 = h_odom(X)
    # bias = getbias(z1,z2)
    zc = np.array([*z1, *z2]).ravel()
    zret = list(zc)
    return zret


def f_differential_drive(X, dt, noise=np.zeros(11), u=None, linear=False):
    """Dynamics of a differential drive robot, this is a constant velocity model, so trajectories are circle segments.
    when the rotational speed is sufficiently low, the system dynamics are approximated using a linearization
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

    X = np.array(X, np.float64)
    Xk1 = np.zeros_like(X)
    if u is not None:
        # Velocity controlled inputs are assumed to directly apply to the velocities.
        vels = utov(u)
        Xk1[3], Xk1[4] = vels
        X[3], X[4] = vels
    else:
        Xk1[3] = X[3]
        Xk1[4] = X[4]
    if np.abs(X[4] * dt) > 0.00001 and not linear:
        logger.debug("Moving along a circular trajectory")
        R = X[3] / X[4]
        dx = R * (np.sin(X[2] + X[4] * dt) - np.sin(X[2]))
        dy = -R * (np.cos(X[2] + X[4] * dt) - np.cos(X[2]))
        Xk1[0] = X[0] + dx
        Xk1[1] = X[1] + dy
        dth = X[4]
    else:
        logger.debug("Moving along a linearized trajectory")
        dx = X[3] * np.cos(X[2]) * dt
        dy = X[3] * np.sin(X[2]) * dt
        Xk1[0] = X[0] + dx
        Xk1[1] = X[1] + dy
        Xk1[4] = X[4]
    deltath = Xk1[4] * dt
    Xk1[2] = X[2] + deltath
    return Xk1


def f_bias(X, dt, noise=np.zeros(11), u=None, linear=False):
    """Dynamics of the filter, with constant velocity biases added"""
    X = np.array(X)
    Xnew = np.zeros_like(X)
    Xsys = f_differential_drive(X[:5], dt, noise, u, linear)
    Xnew[:5] = Xsys

    Xnew[0] = X[0] + np.cos(X[2]) * X[3] * dt
    Xnew[1] = X[1] - np.sin(X[2]) * X[3] * dt
    Xnew[2] = X[2] + X[4] * dt

    bvel = X[8:11]
    # bvel = bvel*0
    Xnew[5] = X[5] + dt * bvel[0]
    Xnew[6] = X[6] + dt * bvel[1]
    Xnew[7] = X[7] + dt * bvel[2]
    Xnew[8] = X[8]
    Xnew[9] = X[9]
    Xnew[10] = X[10]
    return Xnew + noise


class ukf_bias(filterpy.kalman.UnscentedKalmanFilter):
    def __init__(self, *args, **kw):
        super(ukf_bias, self).__init__(*args, **kw)
        self.time = time.perf_counter()

    # Monkey patch the ukf class with convenience functions:
    def predict_rt(self, **kw):
        self.timeNew = time.perf_counter()
        delta = self.timeNew - self.time
        self.time = time.perf_counter()
        logger.info(f"Taking a timestep of {delta}")
        kw.update({"dt": delta})
        self.predict(**kw)


def h_planebots(x):
    return x[:3]


def h_vision(x):
    return x[:3]


# sigmas_bias = MerweScaledSigmaPoints(11, alpha=.1, beta=2., kappa=0)


# ukf = filterpy.kalman.UnscentedKalmanFilter(dim_x=8,
#                                               dim_z=3,
#                                               dt=0.1,
#                                               hx==lambda x: [x[3], x[4]],
#                                               fx=lambda x,dt,u: f_differential_drive(x,u=u,dt=dt),
#                                               points=sigmas,
#                                               sqrt_fn=None,
#                                               x_mean_fn=None,
#                                               z_mean_fn=None,
#                                               residual_x=None,
#                                               residual_z=None)
#
sigmas = MerweScaledSigmaPoints(5, alpha=.1, beta=2., kappa=1.)


class ukFilter(filterpy.kalman.UnscentedKalmanFilter):
    def __init__(self):
        super().__init__(dim_x=5,
                         dim_z=3,
                         dt=0.01,
                         hx=lambda x: [x[3], x[4]],
                         fx=lambda x, dt, u=None: dwa.motion(x, u=u, dt=dt),
                         points=sigmas,
                         sqrt_fn=None,
                         x_mean_fn=None,
                         z_mean_fn=None,
                         residual_x=residual_x,
                         residual_z=residual_z)
        self.he = lambda x: [x[3], x[4]]
        self.hv = h_planebots
        self.Q = np.diag([0.053, 0.053, 0.005, 1, 1])
        self.P = np.eye(5) * 0 + self.Q
        self.tV = time.perf_counter()
        self.tE = time.perf_counter()
        self.tL = time.perf_counter()
        self.Rv = np.eye(3) * [1, 1, 1] * 1E-4
        # self.Re = np.eye(3)*[1,1,1]*1E-4
        self.Re = np.eye(2) * [1, 1] * 1E-4

    def stepplanebots(self, z, mm=True):
        if mm:
            zv = [z[0] / 1000, z[1] / 1000, z[2]]
        else:
            zv = z
        dt = time.perf_counter() - self.tL

        self.predict(u=None, dt=dt)
        self.update(zv, hx=self.hv, R=self.Rv)
        self.tV = self.tL = time.perf_counter()

    def stepElisa(self, z, dt=0):

        # if not dt:
        #     dt = time.perf_counter()- self.tL

        self.predict(u=None, dt=dt)
        self.update(z, hx=self.he, R=self.Re)
        self.tE = self.tL = time.perf_counter()


# ukf.update(z,R = np.eye(2)*[1,1]*1E-4)

if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    logger.addHandler(planebots.log_long)
    ukf = ukf_bias(dim_x=11,
                   dim_z=3,
                   dt=0.1,
                   hx=h_vis,
                   fx=f_bias,
                   points=sigmas,
                   sqrt_fn=None,
                   x_mean_fn=None,
                   z_mean_fn=None,
                   residual_x=residual_x,
                   residual_z=None)

    ukf.x
    for i in range(100):
        ukf.predict_rt()
        ukf.update([1, 1, 1])

    logger.debug("test")
