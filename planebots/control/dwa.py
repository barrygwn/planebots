import configparser
import json
# from planebots import packdir
import logging
import os

# import planebots
from planebots import packdir

# import numpy as np
# from planebots import cvt_2d_sampling
# from scipy import optimize
# import scipy
# from scipy.spatial import Delaunay
# from scipy.spatial import Voronoi
# from scipy.spatial import voronoi_plot_2d
# from scipy import argmin
# from scipy import inner
# import skfmm
# from scipy import interpolate
# import numpy as np
# from ..observers import f_differential_drive
# from planebots.observers import f_differential_drive
# import scipy.ndimage.filters as filters
# import scipy.ndimage.morphology as morphology
# from matplotlib import pyplot as plt
logger = logging.getLogger(__name__)
config = configparser.ConfigParser()
config.read(os.path.join(packdir, 'settings', 'coverage.ini'))
import math
import numpy as np
import matplotlib.pyplot as plt


# from planebots.observers import
class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        # self.max_speed = 1.0  # [m/s]
        # self.min_speed = -0.1  # [m/s]
        # self.max_yawrate = 80.0 * np.pi / 180.0  # [rad/s]
        # self.max_accel = 0.2  # [m/ss]
        # self.max_dyawrate = 80.0 * np.pi / 180.0  # [rad/ss]
        # # self.v_reso = 0.05  # [m/s]
        # self.yawrate_reso = 0.1 * np.pi / 180.0  # [rad/s]
        # self.dt = 0.1  # [s] Time tick for motion prediction
        # self.predict_time = 3.0  # [s]
        # self.to_goal_cost_gain = 1
        # self.speed_cost_gain = 1.0
        # self.obstacle_cost_gain = 1.0
        # self.robot_radius = 0.05  # [m] for collision check
        pass


config = Config()
configini = configparser.ConfigParser()


def domain_plot_coordinates(domain):
    x0, x1, y0, y1 = domain
    xx = [x0, x1, x1, x0, x0]
    yy = [y0, y0, y1, y1, y0]
    return xx, yy


def generate_config(pathname):
    configini = configparser.ConfigParser()
    configini.read(pathname)
    config.dt = json.loads(configini.get("dwa", "dt"))
    config.goal_point_radius = json.loads(configini.get("dwa", "goal_point_radius"))
    config.to_goal_cost_gain = json.loads(configini.get("dwa", "to_goal_cost_gain"))
    config.max_accel = json.loads(configini.get("dwa", "max_accel"))
    config.max_dyawrate = json.loads(configini.get("dwa", "max_dyawrate"))
    config.max_yawrate = json.loads(configini.get("dwa", "max_yawrate"))
    config.max_speed = json.loads(configini.get("dwa", "max_speed"))
    config.min_speed = json.loads(configini.get("dwa", "min_speed"))
    config.predict_time = json.loads(configini.get("dwa", "predict_time"))
    config.speed_cost_gain = json.loads(configini.get("dwa", "speed_cost_gain"))
    config.obstacle_cost_gain = json.loads(configini.get("dwa", "obstacle_cost_gain"))
    config.offpath_gain = json.loads(configini.get("dwa", "offpath_gain"))
    config.robot_radius = json.loads(configini.get("dwa", "robot_radius"))
    config.t_sim = json.loads(configini.get("dwa", "t_sim"))
    config.obstacle_avoidance = configini.getboolean("dwa", "obstacle_avoidance")
    config.v_n = json.loads(configini.get("dwa", "v_n"))
    config.yawrate_n = json.loads(configini.get("dwa", "yawrate_n"))
    return config


def generate_inter_config(pathname=os.path.join(packdir, 'settings', 'dwa_intermediate.ini')):
    configini = configparser.ConfigParser()
    configini.read(pathname)
    config.max_accel = json.loads(configini.get("dwa", "max_accel"))
    config.max_dyawrate = json.loads(configini.get("dwa", "max_dyawrate"))
    config.max_yawrate = json.loads(configini.get("dwa", "max_yawrate"))
    config.max_speed = json.loads(configini.get("dwa", "max_speed"))
    config.min_speed = json.loads(configini.get("dwa", "min_speed"))
    config.v_n = json.loads(configini.get("dwa", "v_n"))
    config.yawrate_n = json.loads(configini.get("dwa", "yawrate_n"))
    config.domain = json.loads(configini.get("dwa", "domain"))
    config.t_safe = json.loads(configini.get("dwa", "t_safe"))
    return config


def reload_config():
    config = generate_config(os.path.join(packdir, 'settings', 'coverage.ini'))
    return config


config = reload_config()
interconfig = generate_inter_config()


def motion(x, u=None, dt=0, linear=False):
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
    if u is not None:
        xacc = [x[0], x[1], x[2], u[0], u[1]]
    else:
        xacc = x.copy()
    xnew = f_differential_drive(xacc, dt=dt, linear=linear)
    xnew[1] = xnew[1]
    xnew[2] = xnew[2]
    xnew[4] = xnew[4]
    return xnew


def f_differential_drive(X, dt, linear=False):
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
        dxacc = 1
        dyacc = Xk1[4] * dt
        # dx = dxacc * np.cos(X[2]) - dyacc *np.sin(X[2])
        # dy = dyacc * np.sin(X[2]) + dyacc*np.cos(X[2])

        dx = X[3] * np.cos(X[2]) * dt
        dy = X[3] * np.sin(X[2]) * dt
        Xk1[0] = X[0] + dx
        Xk1[1] = X[1] + dy
        Xk1[4] = X[4]
    deltath = Xk1[4] * dt
    Xk1[2] = X[2] + deltath
    return Xk1


def window(x, config):
    Vspec = [config.min_speed, config.max_speed,
             -config.max_yawrate, config.max_yawrate]

    logger.debug(f"Vspec {Vspec}")
    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_dyawrate * config.dt,
          x[4] + config.max_dyawrate * config.dt]
    logger.debug(f"Vd {Vd}")

    #  [vmin,vmax, yawrate min, yawrate max]
    dw = [max(Vspec[0], Vd[0]), min(Vspec[1], Vd[1]),
          max(Vspec[2], Vd[2]), min(Vspec[3], Vd[3])]
    logger.debug(f"dw {dw}")
    return dw


# def dwa_calc_input(x,u,dw,config,goal,ob):
#     logger.debug("For every combination of speeds:")
#     # evaluate all trajectory with sampled input in dynamic window
#     for v in np.arange(dw[0], dw[1], config.v_reso):
#         for y in np.arange(dw[2], dw[3], config.yawrate_reso):
#             xn1 = dwa_moveinaxes(x,u,config.predict_time)
#             goal_cost = dwa_goal_cost(x,goal)
#             speed_cost = config.max_speed - xn1[3]
#             obstacle_cost = dwa_calc_obst(x,goal,config)

def dwa_calc_point_collision(x, point, radiusX, radiusPoint):
    """

    :param x:
    :param obstacles:
    :param config:
    :return:
    """
    # Calculate the collision time for a certain vector with points
    vau, omega = x[3:5]
    # Calculate if the two circles intersect.
    R = omega / vau
    cx, cy = cxy1 = np.array([np.sin(x[2]), np.cos(x[2])]) * R - x[:2]
    cxy2 = point
    # Cases: we go around the object, we have a collision, or the point is sufficiently far away
    distance = np.linalg.norm(cxy1 - cxy2)
    if radiusPoint == 0 and radiusX == 0:
        return dwa_line_coll(x, point, point)

    if distance < (radiusX + radiusPoint - R):
        logger.debug("We move around the point, no collision")
    elif distance > (radiusX + radiusPoint + R):
        logger.debug("We are sufficiently far away, no collision")
    else:
        logger.debug("Possible point collision....")
        logger.debug("Orthogonal line between the two circles")
        c = distance
        b = radiusPoint + radiusX
        a = abs(R)
        alpha = np.arccos(-(a ** 2 - b ** 2 - c ** 2) / (2 * b * c))
        c_alpha = b * np.cos(alpha)
        beta = a * np.arccos((c - c_alpha) / c)
        cdelta = cxy2 - cxy1
        phi = np.arctan2(*cdelta[1::-1])
        p1 = cxy1 + c * np.array([np.cos(phi + beta), np.sin(phi + beta)])
        p2 = cxy1 + c * np.array([np.cos(phi - beta), np.sin(phi - beta)])
        return p1, p2
    return


def inDomainTime(x, domain):
    """Returns the tima and """
    x0, x1, y0, y1 = domain
    d1, t1 = dwa_line_coll(x, [x0, y0], [x0, y1])
    d2, t2 = dwa_line_coll(x, [x0, y1], [x1, y1])
    d3, t3 = dwa_line_coll(x, [x1, y1], [x1, y0])
    d4, t4 = dwa_line_coll(x, [x1, y0], [x0, y0])
    t_col = np.min([t1, t2, t3, t4])
    d_col = t_col * x[3]
    return d_col, t_col


def dwa_line_coll(x, p1, p2, approx_straight=True):
    """Calculates the time it takes for an agent with differential drive dynamics to collide with a line between
    p1 and p2"""
    x = np.array(x, np.float64)
    logger.debug(f"Starting point {x[:3]}")
    if x[3] == 0:
        return np.array(np.inf), np.array(np.inf)
    A0 = x[:2]  #
    Av = np.array([np.cos(x[2]), np.sin(x[2])])
    Apn = np.array([-Av[1], Av[0]])  # orthoganal vector

    # Radius of the curvature
    if approx_straight:
        if abs(x[4]) < 1e-6:
            x[4] = 1e-6
    if x[4] != 0:
        R = abs(x[3] / x[4])
        if x[3] * x[4] > 0:  # Determine direction of agent:
            leftcircle = True
            if x[3] > 0:
                clockwise = True
            else:
                clockwise = False
            ctr = A0 + R * Apn
        else:
            leftcircle = False
            if x[3] > 0:
                clockwise = False
            else:
                clockwise = True
            ctr = A0 - R * Apn
        logger.debug(f"Radius is {R}")

        logger.debug(f"Center is {ctr}")
        if p1[0] == p2[0] and p1[1] == p2[1]:
            # Collision with a line
            p = p1 - ctr
            R_acc = np.linalg.norm(p)
            if R_acc - R < 1e4:
                angle = np.arctan2(*p)
                a0 = np.mod(x[2] - angle, 2 * np.pi)
                dist = abs(a0 / x[4] * x[3])
                logger.debug("Point on trajectory")
                time = dist / x[3]
                return np.array(dist), np.array(time)
        # https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm
        # f = B0- ctr
        L = np.array(p2)
        E = np.array(p1)
        C = np.array(ctr)
        r = R
        d = L - E
        f = E - C
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - R ** 2
        discriminant = (b ** 2 - 4 * a * c) ** 0.5

        if np.real(discriminant) - discriminant < 1e8 and discriminant > 1E-4:
            logger.debug("Determinant is real")
            t1 = (-b - discriminant) / (2 * a)
            t2 = (-b + discriminant) / (2 * a)
            P1 = E + t1 * d
            P2 = E + t2 * d
            # p1 = B0 + d*t1
            # p2 = B0 + d*t2
            logger.debug(f"Collision on {p1}{p2}")
            cs = np.vstack((P1, P2))
            XYs = (cs - ctr) / R
            # angles = np.arctan2(XYs[0,:],XYs[1,:])
            vA = x[:2] - ctr
            ang1 = np.arctan2(np.cross(XYs[0], vA), np.dot(XYs[0], vA)) * 180 / np.pi
            # ang113 = np.arctan2(np.cross(Apn,XYs[0]),np.dot(Apn,XYs[0]))*180/np.pi
            ang2 = np.arctan2(np.cross(XYs[1], vA), np.dot(XYs[1], vA)) * 180 / np.pi
            angles = np.arctan2(*XYs.T[::-1])
            if not clockwise:
                ang1m = np.mod(ang1 + 360, 360)
                ang2m = np.mod(ang2 + 360, 360)
            else:
                ang1m = np.mod(-ang1 + 360, 360)
                ang2m = np.mod(-ang2 + 360, 360)
            if 0 < t1 and t1 < 1 and 0 < t2 and t2 < 1:
                # a_start = anglen
                # check circle line distance  -----o------->
                anglep = 0.5  # Compensation for travelling perpendicular to line
                # firstcoln2 = np.mod(-1*at1m+1,1)
                # firstcoln4 = np.mod(-1*at2m+1,1)
                if x[3] > 0:
                    firstcolt12 = min(ang1m, ang2m) / 180 * np.pi
                else:
                    # firstcolt12 = (1- maxanglen)*2*np.pi
                    firstcolt12 = min(ang1m, ang2m) / 180 * np.pi
                dist = abs(firstcolt12 * R)
                timev = abs(firstcolt12 / x[4])
                # Dist is the radius angle travelled??
            # Pierce
            elif 0 <= t1 and t1 <= 1:
                logger.debug("Pierce through")
                firstcolt1 = ang1m * np.pi / 180
                dist = abs(firstcolt1 * R)
                timev = abs(firstcolt1 / x[4])

            elif 0 <= t2 and t2 <= 1:
                firstcolt2 = ang2m * np.pi / 180
                dist = abs(firstcolt2 * R)
                timev = abs(firstcolt2 / x[4])
            else:
                dist = np.inf
                timev = abs(2 * np.pi / x[4])
        else:
            logger.debug("Determinant not real")
            #        Whole circle not on line path
            dist = np.inf
            # firstcol = 2*np.pi
            timev = abs(2 * np.pi / x[4])
        logger.debug("Collision detection performed")
        return np.array(dist), np.array(timev)


def dwa_goal_cost(x, goal):
    dx, dy = goal - x[:2]
    goal_angle = math.atan2(dy, dx)
    pose_angle = x[2]
    cost = abs(goal_angle - pose_angle) % (np.pi)
    return cost


def plot_dw(x, config, ax):
    """Plots the dw in orange"""
    dw = calc_dynamic_window(x, config)
    n = 5
    sp = 5
    vs = np.linspace(dw[0], dw[1], sp)
    os = np.linspace(dw[2], dw[3], sp)
    pts = np.zeros((2, 4 * sp), np.float32)
    i = 0
    for v in reversed(vs):
        p1 = motion([*x[:3], v, dw[2]], dt=config.predict_time)[:2]
        pts[:, i] = p1
        i += 1
    for o in os:
        p1 = motion([*x[:3], dw[0], o], dt=config.predict_time)[:2]
        pts[:, i] = p1
        i += 1
    for v in vs:
        p1 = motion([*x[:3], v, dw[3]], dt=config.predict_time)[:2]
        pts[:, i] = p1
        i += 1
    for o in reversed(os):
        p1 = motion([*x[:3], dw[1], o], dt=config.predict_time)[:2]
        pts[:, i] = p1
        i += 1
    x = pts[0, :]
    y = pts[1, :]
    # ax.plot(x,y,'x')
    ax.plot(x, y, '-.g')
    # ax.plot(*getPath([*x[:3],dw[0],dw[2]],config.predict_time,n),'-.g')
    # ax.plot(*getPath([*x[:3],dw[1],dw[2]],config.predict_time,n),'-.g')
    # ax.plot(*getPath([*x[:3],dw[0],dw[3]],config.predict_time,n),'-.g')
    # ax.plot(*getPath([*x[:3],dw[1],dw[3]],config.predict_time,n),'-.g')


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yawrate, config.max_yawrate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_dyawrate * config.dt,
          x[4] + config.max_dyawrate * config.dt]

    #  [vmin,vmax, yawrate min, yawrate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def calc_obstacle_cost(trajectory, ob, config):
    """
        calc obstacle cost inf: collision
    """
    minval = float("Inf")
    for p in trajectory:
        diff = np.linalg.norm(p - ob, 2, 1)
        minval = min(np.min(diff), minval)
        if (minval <= config.robot_radius):
            return float("Inf")
    return 1.0 / minval  # OK


def calc_to_reference_trajectory(states, u, ref):
    goal = np.array(ref).T
    min_cost = float(1e3)  # Large constant instead of inf to ensure working of obj fcn
    min_segment = -1
    min_coord = np.ones(2) * np.nan
    min_time = np.inf
    logger.debug("Finding collision time")
    min_total_time = float(1e3)  # Large constant instead of inf to ensure working of obj fcn
    for i in range(len(goal) - 1):
        l0 = [goal[i, 0], goal[i, 1]]
        l1 = [goal[i + 1, 0], goal[i + 1, 1]]
        # Time before hitting the line segment:
        x_predict = [*states[:3], *u]
        ans = dwa_line_coll(x_predict, l0, l1)
        distance, timedur = ans[0], float(ans[1].ravel())
        if timedur >= float(1e3):
            # No intersection found
            pass
        else:
            # Intersection found
            time_to_line = timedur
            # Adding travel time when the remainder of the path is followed:
            collision_coord = motion(states, dt=timedur, u=[u[0], u[1]])
            partial_line_distance = np.linalg.norm(l1 - collision_coord[:2])
            stepsz = np.linalg.norm(np.array(l1) - l0)
            remainder = stepsz * (len(goal) - i)
            line_travel_time = (partial_line_distance + remainder) / (abs(u[0]))
            total_time = line_travel_time + time_to_line
            if total_time < min_time:
                # This is the fastest route combination
                min_time = total_time
                min_segment = i
                min_coord = collision_coord[:3]

        return min_cost, min_segment, min_time


def getPath(x, dt, n, linear=True):
    n = int(n)
    times = np.linspace(0, dt, n)
    xx = np.zeros_like(times)
    yy = np.zeros_like(times)
    if linear:
        for k in range(n):
            xn = motion(x, dt=times[1], linear=linear)
            xx[k] = xn[0]
            yy[k] = xn[1]
            x = xn
        return xx, yy
    else:
        for k in range(n):
            xn = motion(x, dt=times[k], linear=False)
            xx[k] = xn[0]
            yy[k] = xn[1]
        return xx, yy


def heading_angle(goal, states, config):
    """Calc the angle with the closest point of the goal"""
    firstPoint = goal[:, 0]
    ratio = 1
    tipstates = motion(states, dt=config.predict_time / ratio, linear=False)
    xx, yy = firstPoint - tipstates[:2]
    angle1 = math.atan2(yy, xx)
    if tipstates[3] > 0:
        angle2 = tipstates[2]

    else:
        angle2 = tipstates[2] + np.pi
    dangle = abs(remap_angle(angle1 - angle2))
    dist = np.linalg.norm(tipstates[:2] - firstPoint) * 10
    return dangle + dist


def remap_angle(x):
    """maps the angle onto the domain [-pi, pi) """
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    if x > np.pi:  # move to [-pi, pi)
        x -= 2 * np.pi
    return x


# def heading(goal,states,config):
#     """
#         calc to goal cost with angle difference
#     """
#     if len(goal.T) == 2 and len(goal) ==1: # We only have an end point as goal
#
#         dx = goal[0] - trajectory[-1, 0]
#         dy = goal[1] - trajectory[-1, 1]
#         error_angle = math.atan2(dy, dx)
#         cost_angle = error_angle - trajectory[-1, 2]
#
#
#         cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
#         cost = np.linalg.norm([dx,dy])
#         return cost
#     elif len(goal.T) == 2:
#         #
#         goal = np.array(goal).T
#
#     min_cost = float(1e3) #Large constant instead of inf to ensure working of obj fcn
#     logger.debug("Finding collision time")
#
#     times, collision_point, idx = toLineTime(states,goal)
#     if states[3] > 1e-6: # if we are not standing still
#         if idx == -1 or times > config.predict_time : #if no collision or collision outside timeframe:
#             # No collision with path
#             tip = motion(states,dt=config.predict_time)
#             idxc = closestPoint(goal,tip[:2])
#             cp = goal.T[idxc]
#             s = np.linalg.norm(cp-tip[:2])
#             time0 = s/abs(states[3])
#             stepsz = np.linalg.norm(goal.T[0]-goal.T[1])
#             goallen = (np.size(goal)/2-idxc-1)*stepsz/abs(states[3])
#             time0+= goallen
#             time0+= config.predict_time
#             times = time0
#     return times, collision_point, idx
def closestPoint(path, point):
    if np.shape(path)[0] == 2:
        xydistance = path.T - point
    else:
        xydistance = path - point
    absdistance = np.linalg.norm(xydistance, axis=1)
    mindistance = np.min(absdistance)
    idxs = np.where(absdistance == mindistance)
    return int(idxs[0])


def toLineTime(states, trajectory):
    """Calculates the time before intersecting the trajectory
    Returns: times, collision_point, idx"""
    goal = trajectory
    if len(goal.T) == 2:  # We only have an end point as goal
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]

        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        cost = np.linalg.norm([dx, dy])
        return cost
    else:
        #
        goal = np.array(goal).T
        min_cost = float(1e6)  # Large constant instead of inf to ensure working of obj fcn
        logger.debug("Finding collision time")
        min_total_time = float(1e6)  # Large constant instead of inf to ensure working of obj fcn
        min_idx = -1
        min_coll = [0, 0]
        for i in range(len(goal) - 1):
            l0 = [goal[i, 0], goal[i, 1]]
            l1 = [goal[i + 1, 0], goal[i + 1, 1]]
            # Time before hitting the line segment:
            x_predict = states
            ans = dwa_line_coll(x_predict, l0, l1)
            distance, timedur = ans[0], float(ans[1].ravel())
            if distance >= float(1e3):
                # No intersection found
                continue
            else:
                # Intersection found.
                time_to_line_plain = timedur
                time_to_line = time_to_line_plain * config.offpath_gain
                # Adding travel time when the remainder of the path is followed:
                collision_coord = motion(states, dt=timedur)
                partial_line_distance = np.linalg.norm(l1 - collision_coord[:2])
                stepsz = np.linalg.norm(np.array(l1) - l0)
                remainder = stepsz * (max(len(goal) - i - 2, 0))  # head of the path
                line_travel_time = (partial_line_distance + remainder) / (abs(states[3]))
                total_time = line_travel_time + time_to_line
                if total_time < min_total_time:
                    # This is the fastest route combination
                    min_total_time = total_time
                    min_idx = i
                    min_coll = collision_coord
                    # minvw = np.array([s,o])
                    #
                    # logger.info(f"{l0[0]:4.2f} {l1[0]:4.2f} {s:4.2f}|{o:4.2f} --> {total_time:4.2f}")
        return min_total_time, min_coll, min_idx

        #
        # if min_cost >= float(1e3):
        #     #See where the
        #     trajectory_end = trajectory[-1, :2]
        #     goal_end = goal[-1, :2]
        #     xydistance = goal - trajectory_end
        #     absdistance = np.linalg.norm(xydistance,axis=1)
        #     mindistance = np.min(absdistance)
        #     idxs = np.where(absdistance==mindistance)
        #
        #     combined_cost = mindistance+min_cost
        #     return combined_cost
        # else:
        #     return min_cost


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    traj = np.array(x)
    time = 0
    while time <= config.predict_time:
        xnew = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, xnew))
        time += config.dt
        x = xnew
    return traj


def calc_filter_input(x, dw, u_in, config):
    """"""
    domain = config.domain
    d_col, t_col = inDomainTime([*x[:3], *u_in], domain)
    if t_col > config.t_safe:
        return u_in, 0
    else:
        best_cost = 1e6
        best_u_in = [0, 0]
        vi, wi = u_in
        vmax, wmax = config.max_speed, config.max_yawrate
        for w in np.linspace(dw[2], dw[3], config.yawrate_n):
            for v in np.linspace(dw[0], dw[1], config.v_n):
                x_in = [*x[:3], v, w]
                d_col2, t_col2 = inDomainTime(x_in, domain)
                if t_col2 > config.t_safe:
                    cost = (vi - v) ** 2 / vmax ** 2 + (wi - w) ** 2 / wmax ** 2
                    if cost < best_cost:
                        best_cost = cost
                        best_u_in = [v, w]

        return best_u_in, best_cost


def calc_filter_input_cont(x, dw, u_in, config):
    """"""
    domain = config.domain
    x0, x1, y0, y1 = domain
    d_col, t_col = inDomainTime([*x[:3], *u_in], domain)
    if x[0] < x0 or x1 < x[0] or x[1] < y0 or y1 < x[1]:
        # Outside the domain
        return [0, 0], 0
    elif t_col > config.t_safe:
        return u_in, 0
    else:
        cost = t_col / config.t_safe
        u_out = np.array(u_in) * t_col / config.t_safe

        return u_out, cost


def calc_final_input(x, dw, goal, ob=[], config=configini):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    # evaluate all trajectories with sampled input in dynamic window
    ntraj = int(np.round(config.predict_time // config.dt))
    traj = np.zeros((ntraj, 2), np.float32)
    for y in np.linspace(dw[2], dw[3], config.yawrate_n):
        for v in np.linspace(dw[0], dw[1], config.v_n):
            # trajectory = predict_trajectory(x_init, v, y, config)

            # calc costs
            gamma = config.speed_cost_gain
            alpha = config.to_goal_cost_gain
            beta = config.obstacle_cost_gain
            velo = -abs(v)
            # headingt, collision_point, idx = heading(goal,[*x[:3],v,y],config)
            # headings = headingt*abs(v)
            headingv = heading_angle(goal, [*x[:3], v, y], config)
            speed_cost = gamma * velo * 100 * 0
            if (ob is not None):
                if len(ob) and config.obstacle_avoidance:
                    xx, yy = getPath([*x[:3], v, y], config.predict_time, ntraj, linear=True)
                    traj[:, 0] = xx
                    traj[:, 1] = yy
                    dist = calc_obstacle_cost(traj, ob, config)
                else:
                    dist = 0
            else:
                dist = 0
            ch = alpha * headingv
            cv = beta * dist
            cd = gamma * velo
            final_cost = ch + cv + speed_cost + cd

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                # best_trajectory = trajectory
    return best_u, min_cost


def dwa_intermediate(x, u_in, config):
    """Filters the input for admissable inputs."""
    dw = calc_dynamic_window(x, config)
    u, min_cost = calc_filter_input(x, dw, u_in, config)


def dwa_control(x, config, goal, ob=None):
    """
        Dynamic Window Approach control
    """

    dw = calc_dynamic_window(x, config)

    u, min_cost = calc_final_input(x, dw, goal, ob, config)

    return u, min_cost


def plot_arrow(x, y, yaw, length=0.1, width=0.05):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_agent(x, config, ax, n=10):
    circlex, circley = getPath([x[0], x[1] - config.robot_radius, 0, config.robot_radius, 1], 2 * np.pi, n,
                               linear=False)
    ax.plot(circlex, circley, '--g')
    circlex, circley = getPath([x[0], x[1] - config.goal_point_radius, 0, config.goal_point_radius, 1], 2 * np.pi, n,
                               linear=False)
    ax.plot(circlex, circley, '--r')


if __name__ == '__main__':
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    n_path = 10
    fmmpath = np.array([np.linspace(-8, 2, n_path) / 10, np.linspace(-8, 2, n_path) / 10])
    r = 0.4

    npt = 30
    circlex, circley = getPath([0, -r, 0, r, 1], 2 * np.pi, npt)
    noise = np.random.rand(2, npt) * 0.02
    fmmpath = np.array([-circlex + noise[0], -circley + noise[1]])
    x = np.array([0.5, 0.8, np.pi * 1.5, 0, 0])

    # Make a line to follow
    line = np.linspace(10, 2, 10)
    # Find fastest way to line:
    speeds = np.linspace(0, 1, 10)
    omegas = np.linspace(0, 1, 10)
    min_time = np.inf
    minvw = np.array([0, 0])
    states = x

    for i in range(len(line) - 1):
        l0 = [line[i], line[i]]
        l1 = [line[i + 1], line[i + 1]]
        for s in speeds:
            for o in omegas:
                states[3] = s
                states[4] = o
                ans = dwa_line_coll(states, l0, l1)
                distance, timedur = ans[0], float(ans[1].ravel())
                if timedur < min_time:
                    min_time = timedur
                    minvw = np.array([s, o])
                    logger.info(f"{l0[0]:4.2f} {l1[0]:4.2f} {s:4.2f}|{o:4.2f} --> {min_time:4.2f}")

    show_animation = True
    fig, ax = plt.subplots(figsize=(10, 10))
    pathor = fmmpath.copy()
    # for i in range(100):
    traj = x
    t_tot = 0
    while True:
        u, predicted_trajectory = dwa_control(x, config, fmmpath)
        t_tot += config.dt
        x = motion(x, u, config.dt)  # simulate robot
        traj = np.vstack((traj, x))
        target = fmmpath[:, 0]
        circlex, circley = getPath([target[0], target[1] - config.robot_radius, 0, config.robot_radius, 1], 2 * np.pi,
                                   30)
        dist_to_goal = math.sqrt((x[0] - target[0]) ** 2 + (x[1] - target[1]) ** 2)
        # check reaching goal
        if show_animation:
            plt.cla()
            plt.axis("equal")
            plot_agent(x, config, plt)
            plt.plot(*getPath([*x[:3], *u], config.predict_time, 10), "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(pathor[0, :], pathor[1, :], "-.b")
            plt.plot(fmmpath[0, :], fmmpath[1, :], "-xb")
            plt.plot(traj[:, 0], traj[:, 1], "-k")
            plot_dw(x, config, plt)
            plot_arrow(x[0], x[1], x[2])
            ax.set_ylim(-2, 2)
            ax.set_xlim(-2, 2)
            ax.grid(True)

            ax.text(x[0], x[1], f"   t{t_tot:4.2f}|dist{dist_to_goal:4.2f}{i} a{x[2]:4.2f}v{x[3]:4.2f},w{x[4]:4.2f}")
            plt.pause(0.0001)

        if dist_to_goal <= config.goal_point_radius:
            l = len(fmmpath[0])
            if l > 1:
                logger.debug(f"{l}")
                print(f"{l} Goal!!")
                fmmpath = fmmpath[:, 1:]
            else:
                break
            # break

    logger.debug("Done")

    #
    # ax.plot(fmmpath[:,0],fmmpath[:,1])
    # g = 1
    #
    # x = np.array([0,0,0,1,0.3])
    # m12,m2,m3 = calc_to_reference_trajectory(x,u=[1,1],ref=fmmpath)
    # traj = np.zeros((10,5))
    # for k in range(10):
    #     dt= k/10*m3
    #     xn = motion(x,u=x[3:5],dt=dt)
    #     traj[k,:] = xn
    # ax.plot(traj[:,0],traj[:,1])
    # dist,time = dwa_line_coll(x,[0,0.5],[1,0.5])
    # ts =float(time.ravel())
    # # ax.text(i/g,j/g,f"{ts:4.2f}")
    # # plt.show()
    # # cal

    # fig, ax = plt.subplots()
    # xdata, ydata = [], []
    # path, = plt.plot([], [], 'ro')
    # ln, = plt.plot([], [], 'ro')
    #
    # def init():
    #     # ax.set_xlim(0, 2*np.pi)
    #     # ax.set_ylim(-1, 1)
    #     ax.set_ylim(-2,2)
    #     ax.set_xlim(-2,2)
    #     return path,ln,
    #
    # def update(frame):
    #     xx = frame[0][0]
    #     yy = frame[0][1]
    #     xdata.append(xx)
    #     ydata.append(yy)
    #
    #     ln.set_data(xdata, ydata)
    #     path.set_data(frame[1][:,0],frame[1][:,1])
    #     return path,ln,

    # xxx = []
    # pd = []
    # for i in range(100):
    #     u, predicted_trajectory = dwa_control(x, config, goal)
    #     x = motion(x, u, config.dt)  # simulate robot
    #     trajectory = np.vstack((trajectory, x))  # store state history
    #     xxx.append(x)
    #     pd.append(predicted_trajectory)
    #     frames = list(zip(xxx,pd))
    #     update(frames[0])
    #     logger.info(f"{i}")
    #
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.set_ylim(0,1)
    # ax.set_xlim(0,1)
    # ax.plot(np.array(xxx)[:,0],np.array(xxx)[:,1])
    #
    #
    # # ani = FuncAnimation(fig, update, frames=frames,
    # #                     init_func=init, blit=True)
    # plt.show()
# m1,m2,m3 = calc_to_reference_trajectory(x,u=[1,1],ref=fmmpath)
# logger.info(f"{m1}{m2}{m3}")
# m1,m2,m3 = calc_to_reference_trajectory(x,u=[1,-1],ref=fmmpath)
# logger.info(f"{m1}{m2}{m3}")
# m1,m2,m3 = calc_to_reference_trajectory(x,u=[-1,0.1],ref=fmmpath)
# logger.info(f"{m1}{m2}{m3}")
# m1,m2,m3 = calc_to_reference_trajectory(x,u=[-1,-0.1],ref=fmmpath)
# logger.info(f"{m1}{m2}{m3}")
