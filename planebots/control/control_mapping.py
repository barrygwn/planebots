import logging

import numpy as np

from planebots.control import model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import planebots
import configparser
import os

cfg = configparser.ConfigParser()
cfg.read(os.path.join(planebots.packdir, 'settings', 'config.ini'))
potential_kv = cfg.getfloat("potential_field", "kv")
potential_kw = cfg.getfloat("potential_field", "kw")
from planebots.control.observers import remap_angle


# Implementation from the coverage paper:
def map_velocity(fieldvector, angle, kw=potential_kv):
    """Maps the fieldvector to an error e_i, a gain omega_i and an absolute value"""
    theta_i = remap_angle(angle)
    u1, u2 = fieldvector
    theta_di = remap_angle(np.arctan2(u2, u1))
    k_wi = kw
    if (theta_i - theta_di) <= -np.pi:
        omega_i = k_wi * (theta_di - theta_i - 2 * np.pi)
        e_ti = (theta_di - theta_i - 2 * np.pi)
    if -np.pi < (theta_i - theta_di) and (theta_i - theta_di) <= np.pi:
        omega_i = k_wi * (theta_di - theta_i)
        e_ti = theta_di - theta_i
    if np.pi < (theta_i - theta_di):
        omega_i = k_wi * (theta_di - theta_i + 2 * np.pi)
        e_ti = (theta_di - theta_i + 2 * np.pi)
    v_abs = np.linalg.norm(fieldvector)
    return e_ti, v_abs


from planebots.gui import overlays
from planebots.vision import detection


def controlfield(x, field, domain=detection.field_size_mm):
    norm1 = np.array((x[0], x[1])) / domain
    zx, zy = field(*norm1)
    e_ti, v_abs = map_velocity([zx, zy], x[2])
    vabs = np.linalg.norm([zx, zy])
    return e_ti, vabs


if __name__ == '__main__':
    from planebots import dwa

    logger.addHandler(logging.StreamHandler())

    norm1 = np.array((100, 100)) / detection.field_size_mm

    zx, zy = overlays.rotationalfield(*norm1)

    e_ti, v_abs = controlfield([0, 0, 0, 0, 0], overlays.rotationalfield, detection.field_size_mm)
    e_ti, v_abs = controlfield([0, 500, 0, 0, 0], overlays.rotationalfield, detection.field_size_mm)
    e_ti, v_abs = controlfield([400, 0, 0, 0, 0], overlays.rotationalfield, detection.field_size_mm)
    e_ti, v_abs = controlfield([400, 500, 0, 0, 0], overlays.rotationalfield, detection.field_size_mm)

    # omega_i, e_ti, zx, zy = controlfield([100,100,0],overlays.rotationalfield)
    # omega_i, e_ti = map_velocity([zx, zy],0)
    # logger.info(f"omega_i={omega_i} e_ti={e_ti}")

    agent = model.Agent([0.05, 0, -np.pi / 8, 0, 0])


    def linfield(x, y):
        return np.array([x * 0 - 1, y * 0])


    n = 100
    states = np.zeros((n, 5), np.float)
    states2 = np.zeros((n, 5), np.float)

    x0, x1, y0, y1 = domain = [0, 0.5, 0, 0.5]
    xs = x = [0.45, 0.4, 0, 0, 0]
    # x0 = x = [0,0,0,0,0]
    ke = 1
    for i in range(n):
        e_ti, v_abs = controlfield(x, overlays.rotationalfield, domain=[x1, y1])

        v = 0.4 * v_abs * ke
        v = 0.75 * ke
        w = 25 * e_ti * ke
        xn = dwa.motion(x, u=[v, w], dt=0.05)
        # xn = [0.1,i/n,0,0,0]
        states[i] = xn
        x = xn
    xs = x = [0, 0.5, 0, 0, 0]
    for i in range(n):
        e_ti, v_abs = controlfield(x, overlays.rotationalfield, domain=[x1, y1])
        # ke = 4
        v = 0.4 * v_abs * ke
        v = 0.75 * ke
        w = 25 * e_ti * ke
        xn = dwa.motion(x, u=[v, w], dt=0.05)
        # xn = [0.1,i/n,0,0,0]
        states2[i] = xn
        x = xn

    # for i in range(10):
    #     agent.move(l,r,0.1)
    import matplotlib.pyplot as plt

    logger.info("Save a png of the field")
    x, y = np.meshgrid(np.linspace(0, 0.5, 10), np.linspace(0, 0.5, 10))
    ZU, ZV = overlays.rotationalfield(x, y, x_end=x1, y_end=y1)
    # ZU, ZV = linfield(x, y)
    # ZU, ZV = ZU*0+1, ZV*0+1
    speed = np.sqrt(ZU * ZU + ZV * ZV)
    lw = 10 * speed / speed.max()
    lw = 1 * speed / speed.max()
    # ax.quiver(X,Y,ZU,ZV)
    # Since the y-axis points down Y is negative
    fig, ax = plt.subplots()
    ax.streamplot(x, y, ZU, ZV, linewidth=lw, density=[1, 1], color='k')

    ax.plot(*states.T[:2])
    ax.plot(*states2.T[:2])
    plt.show()
