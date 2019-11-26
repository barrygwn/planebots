import logging
import time

import numpy as np

import planebots.gui.overlays as overlays

logger = logging.getLogger(__name__)
from planebots.output import gctronic

import planebots
import inputs
import json

v_scale = json.loads(planebots.config.get("gamepad", "v_scale"))
omega_scale = json.loads(planebots.config.get("gamepad", "omega_scale"))
# Implements controllers for controlling an agent
gamepad_states_default = {
    "ABS_X": 0,
    "ABS_Y": 0,
    "ABS_Z": 0,
    "ABS_RX": 0,
    "ABS_RY": 0,
    "ABS_RZ": 0,
    "ABS_HAT0X": 0,
    "ABS_HAT0Y": 0,
    "ts": 0.0,
    "DX": 0,
    "DY": 0
}
import filterpy
from filterpy.kalman import MerweScaledSigmaPoints

sigmas = MerweScaledSigmaPoints(8, alpha=.1, beta=2., kappa=0)


def h_planebots(x):
    return [x[0], x[3], x[6]]


def h_elisa(x, s=1):
    """measurement function of the elisa robot, measurement of the ticks of the wheels, and the acceleration perpendicular to the motion direction."""
    b = 0.00408
    vau = np.linalg.norm([x[1], x[4]])
    omega = x[7]
    z1 = (vau + (1 / 2) * omega * b) * s
    z2 = (vau - (1 / 2) * omega * b) * s
    z3 = -vau * omega
    return [z1, z2, z3]


ukf_e = filterpy.kalman.UnscentedKalmanFilter(dim_x=8,
                                              dim_z=3,
                                              dt=0.1,
                                              hx=h_elisa,
                                              fx=lambda x, dt, u: f_kalman(x, u=u, dt=dt),
                                              points=sigmas,
                                              sqrt_fn=None,
                                              x_mean_fn=None,
                                              z_mean_fn=None,
                                              residual_x=None,
                                              residual_z=None)
ukf_e.Q = np.diag([1, 0.1, 0.01, 1, 0.1, 0.01, 1, 0.1])
ukf_e.P = np.eye(8) * 10E2
ukf_v = filterpy.kalman.UnscentedKalmanFilter(dim_x=8,
                                              dim_z=3,
                                              dt=0.1,
                                              hx=h_planebots,
                                              fx=lambda x, dt, u: f_kalman(x, u=u, dt=dt),
                                              points=sigmas,
                                              sqrt_fn=None,
                                              x_mean_fn=None,
                                              z_mean_fn=None,
                                              residual_x=None,
                                              residual_z=None)
ukf_v.x = np.zeros((8))
ukf_v.Q = np.eye(8) * .4


def dwa_line_coll(x, p1, p2):
    pts = np.array([[p1], [p2]]).reshape(2, 2)
    B0 = np.matmul(pts.T, np.array([1, 0]))
    Bv = np.matmul(pts.T, np.array([-1, 1]))
    A0 = x[:2]  #
    Av = np.array([np.cos(x[2]), np.sin(x[2])])
    Apn = np.array([-Av[1], Av[0]])  # orthoganal vector

    # Radius of the curvature
    R = abs(x[3] / x[4])
    ctr = A0 + R * Apn
    if p1[0] == p2[0] and p1[1] == p2[1]:
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
    f = B0 - ctr
    d = Bv  # wall direction
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - R ** 2
    disc = (b ** 2 - 4 * a * c) ** 0.5

    if np.real(disc) - disc < 1e8 and disc > 1E-4:
        logger.debug("Determinant is real")
        t1 = (-b - disc) / (2 * a)
        t2 = (-b + disc) / (2 * a)
        p1 = B0 + d * t1
        p2 = B0 + d * t2
        cs = np.vstack((p1, p2))
        XYs = (cs - ctr) / R
        # angles = np.arctan2(XYs[0,:],XYs[1,:])
        angles = np.arctan2(*XYs.T)
        angle0 = x[2]
        if 0 < t1 and t1 < 1 and 0 < t2 and t2 < 1:
            # check circle line distance  -----o------->
            firstcol = min(np.mod(angle0 - angles, 2 * np.pi))
            dist = abs(firstcol / x[4] * x[3])
            # Dist is the radius angle travelled??
        # Pierce
        elif 0 <= t1 and t1 <= 1:
            firstcol = np.mod(angle0 - angles[0], 2 * np.pi)
            dist = abs(firstcol / x[4] * x[3])
        elif 0 <= t2 and t2 <= 1:
            firstcol = np.mod(angle0 - angles[1], 2 * np.pi)
            dist = abs(firstcol / x[4] * x[3])
        else:
            dist = np.inf
            firstcol = 2 * np.pi

    else:
        logger.debug("Determinant not real")
        #        Whole circle not on line path
        dist = np.inf
        firstcol = 2 * np.pi
    time = dist / x[3]

    logger.debug("Collision detection performed")
    return np.array(dist), np.array(time)


def f_kalman(X, u=[0, 0], dt=1):
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

    X = list(np.array(X))
    # x dx ddx y dy ddy th dth
    # x dx ddx y dy ddy th omega
    Xk1 = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    x, dx, ddx, y, dy, ddy, th, omega = X
    rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).squeeze()

    # dxv = np.cos(th)*u[0]
    # dyv = np.sin(th)*u[0]
    dxyv = np.array([X[1], X[4]])
    vauik = np.dot(np.matmul(rm, np.array([1, 0])), dxyv)
    # vaup = np.dot(np.matmul(rm,np.array([0,1])),dxyv)

    # vau = vaui
    # vau, vaup = np.matmul(rm ,np.array([dy,dx])) #Perpendical motion should be constraint --> vaup == 0
    vau, omega = list(u)

    if X[7] < 0.001:
        dxk = X[1] * dt
        dyk = X[4] * dt
        dthk = X[7] * dt
    else:
        dxk = X[1] * dt
        dyk = X[4] * dt
        dthk = X[7] * dt

    if np.abs(omega * dt) > 0.00001:
        # logger.debug("Moving along a circular trajectory")
        R = vau / omega

        dxk1 = R * (np.sin(th + omega * dt) - np.sin(th))
        dyk1 = -R * (np.cos(th + omega * dt) - np.cos(th))
        Xk1[0] = x + dxk
        Xk1[1] = dxk1 / dt
        Xk1[2] = -omega * vau * np.sin(th)
        Xk1[3] = y + dyk
        Xk1[4] = dyk1 / dt
        Xk1[5] = omega * vau * np.cos(th)
        Xk1[6] = (th + dthk)
        Xk1[7] = omega
    else:
        # logger.debug("Moving along a linearized trajectory")
        dx_inc = vau * np.cos(th)
        dy_inc = vau * np.sin(th)
        Xk1[0] = X[0] + dxk
        Xk1[1] = dx_inc
        Xk1[2] = -omega * vau * np.sin(th)
        Xk1[3] = X[3] + dyk
        Xk1[4] = dy_inc
        Xk1[5] = omega * vau * np.cos(th)
        Xk1[6] = (th + dthk)
        Xk1[7] = omega

    # Xk1[1] = dxv
    # Xk1[4] = dyv

    # Xk1[7] = omega
    # if abs(Xk1[6]) > 2*np.pi:
    #     logger.warning("Too high")
    return np.array(Xk1).ravel()


def dwa_moveinaxes(x, u, dt):
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
    vau, omega = u
    X = list(x)
    Xk1 = [0, 0, 0, 0, 0]
    if np.abs(omega * dt) > 0.00001:
        logger.debug("Moving along a circular trajectory")
        R = vau / omega
        dx = R * (np.sin(X[2] + omega * dt) - np.sin(X[2]))
        dy = -R * (np.cos(X[2] + omega * dt) - np.cos(X[2]))
        Xk1[0] = X[0] + dx
        Xk1[1] = X[1] + dy
        dth = omega
        Xk1[4] = dth
    else:
        logger.debug("Moving along a linearized trajectory")
        dx = vau * np.cos(X[2]) * dt
        dy = vau * np.sin(X[2]) * dt
        Xk1[0] = X[0] + dx
        Xk1[1] = X[1] + dy
        Xk1[4] = 0
    deltath = Xk1[4] * dt
    Xk1[2] = (X[2] + deltath + 2 * np.pi) % (2 * np.pi)
    Xk1[3] = vau

    return Xk1


def showGamePadStates(img, gamepad_states, pos=(0, 0), **tlopts):
    items = gamepad_states.items()
    keys = []
    vals = []
    for k, v in items:
        if type(v) == int:
            keys.append(f"{k:10s}")
            vals.append(f":{v:<8.0f}")

        # logger.debug("ksdlf")
    planebots.detection.addTextlines(img, keys, offset=pos, **tlopts)
    planebots.detection.addTextlines(img, vals, offset=(pos[0] + 100, pos[1]), **tlopts)


def monitor_gamepad(gamepad_states):
    logger.debug("Updating the gamepad states")
    events = inputs.get_gamepad()
    for event in events:
        gamepad_states[event.code] = event.state

        try:
            gamepad_states["COUNTER"] += 1
            # Select another agent with top triggers:
            if event.code == 'BTN_TR' and event.ev_type == 'Key' and event.state == 1:
                gamepad_states["previous_id"] = gamepad_states["current_id"]  # Store last entry for setting l,r = 0
                gamepad_states["current_id"] = (gamepad_states["current_id"] + 1) % gamepad_states["n_agents"]
            elif event.code == 'BTN_TL' and event.ev_type == 'Key' and event.state == 1:
                gamepad_states["current_id"] = (gamepad_states["current_id"] - 1) % gamepad_states["n_agents"]
                gamepad_states["current_id"] = (gamepad_states["current_id"] - 1) % gamepad_states["n_agents"]
            elif event.code == 'BTN_EAST' and event.ev_type == 'Key' and event.state == 1:
                gamepad_states["info"] = not gamepad_states["info"]
            try:
                vhat = -gamepad_states["ABS_HAT0Y"] * v_scale
                omegahat = gamepad_states["ABS_HAT0X"] * omega_scale
                if vhat != 0:
                    v = vhat + np.sign(vhat) * gamepad_states["ABS_Z"]
                else:
                    v = gamepad_states["ABS_Z"]
                    v = 0
                if omegahat != 0:
                    omega = omegahat + np.sign(omegahat) * gamepad_states["ABS_Z"]
                else:
                    omega = gamepad_states["ABS_Z"]
                    omega = 0
                l = np.clip((v + omega) // 2, -127, 127)
                r = np.clip((v - omega) // 2, -127, 127)
                # gamepad_states['l'],gamepad_states['r'] = max(min(l,125)-125),max(min(r,125)-125)
                gamepad_states['l'], gamepad_states['r'] = l, r
            except:
                l = 0
                r = 0
                gamepad_states['l'], gamepad_states['r'] = l, r
        except KeyError as e:
            gamepad_states[e.args[0]] = 0
        except Exception as e:
            logger.error(e)

        return gamepad_states


def errors(coords, theta, magXY=np.array([0, 0]), offset=np.array([0, 0]), scale=1,
           vref=10):
    X = coords[0]
    Y = coords[1]
    # e_perp = np.sqrt(+(X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.2
    # fieldvec = field(X, Y)
    # fieldvec = np.array([X,Y])
    fieldvec = magXY
    agentvec = np.array([np.sin(theta), +np.cos(theta)])
    e_theta = angle_between(fieldvec, agentvec)
    # e_theta -=
    e_v = vref - np.linalg.norm(fieldvec)
    return 0, e_theta, e_v


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.linalg.norm(v1) < 0.01 or np.linalg.norm(v2) < 0.01:
        return 0
    # dot = np.dot(v1,v2)
    det = v1[0] * v2[1] - v1[1] * v2[0]
    # return np.arctan2(det,dot)
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.sign(det) * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


if __name__ == '__main__':
    # rotationalfield(x, y, offset_norm=np.array([0, 0]), scaler=6, radius=0.5):
    rFieldOpts = {'offset_norm': np.array([0, 0]), 'scaler': 6, 'x_end': 400, 'y_end': 400}
    logger.addHandler(planebots.log_long)
    logger.setLevel(logging.DEBUG)
    magXY = overlays.rotationalfield(0, 0, **rFieldOpts)

    ang = 0.1
    agentDir = np.array([np.sin(ang), np.cos(ang)])
    # Angle between arrow to right
    a1 = angle_between(magXY, agentDir) * 180 / np.pi
    a2 = angle_between(np.array([0, 1]), magXY) * 180 / np.pi
    kpth = 0.1
    vleft = int(kpth * a1)
    vright = int(-kpth * a1)

    aid = 3655
    Comm = gctronic.Elisa3Comm()

    result2 = Comm.getIdFromAddress(aid)
    batt = Comm.getBatteryAdc(aid)
    Comm.setBlue(aid, 0)
    Comm.setLeftSpeed(aid, vleft)
    Comm.setRightSpeed(aid, vright)
    tic = time.perf_counter()
    Comm.setLeftSpeed(aid, 0)
    Comm.setRightSpeed(aid, 0)

    toc = time.perf_counter() - tic
    logger.info(f"Time elapsed {toc}")
    # time.sleep(1)
    Comm.setBlue(aid, 0)

    ci = 0
    l = 5
    r = -5
    try:
        while True:
            for i in range(Comm.nRobots):
                Address = Comm.AddressList[i]
                if i == ci:
                    # Set the blue LED intensity to the level of the right trigger:
                    # intensity = int(gamepad_states["ABS_RZ"]*100/255)
                    Comm.setLeftSpeed(Address, l)
                    Comm.setRightSpeed(Address, r)
                    # Comm.setBlue(Address,intensity)
                    Comm.setRed(Address, 1)
                else:
                    Comm.setLeftSpeed(Address, 0)
                    Comm.setRightSpeed(Address, 0)
                    # Comm.setBlue(Address,0)
                    Comm.setRed(Address, 0)
                    # Show the communication is live:
                    Comm.setGreen(Address, 1)

            time.sleep(1)
            if ci == Comm.nRobots - 1:
                break
            ci = (ci + 1) % Comm.nRobots


    except (KeyboardInterrupt, SystemExit) as e:
        for i in range(Comm.nRobots):
            Comm.setLeftSpeed(Comm.AddressList[i], 0)
            Comm.setRightSpeed(Comm.AddressList[i], 0)
        Comm.close()
    for i in range(Comm.nRobots):
        Comm.setLeftSpeed(Comm.AddressList[i], 0)
        Comm.setRightSpeed(Comm.AddressList[i], 0)
    time.sleep(0.1)
    Comm.close()

    logger.info("")
