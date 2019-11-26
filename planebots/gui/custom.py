# Custom control law for controlling agents

import time

import numpy as np


def c(j, x, dt):
    periodHz = 0.25
    omega = periodHz * 2 * np.pi
    gain = 10
    u = 0
    t = time.perf_counter()
    w = np.sin(omega * t) * gain

    return [u, w]
