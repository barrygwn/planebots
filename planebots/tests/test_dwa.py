# from vision import coverage_control
import logging
# logger = logging.getLogger(__name__)
import unittest

import matplotlib.pyplot as plt
import numpy as np

# from scipy.spatial import Voronoi, voronoi_plot_2d
# import skfmm
logger = logging.getLogger("vision")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
# import vision
# import datetime
# from scipy import interpolate
# from vision import controllers
# import filterpy
from filterpy.kalman import MerweScaledSigmaPoints

# from filterpy.kalman import UnscentedKalmanFilter as UKF
dt = 0.1
sigmas = MerweScaledSigmaPoints(5, alpha=.1, beta=2., kappa=1.)

from planebots.control import dwa
import time


# import vision.model as model

def tick2thomega(ticks, dt):
    thomega = np.array([0.5 * (ticks[0] + ticks[1]), (ticks[0] - ticks[1]) / 0.0408]) / 170  # wheelbase of 40.8mm
    return thomega


class TestDwa(unittest.TestCase):
    fmmpath = np.array([np.linspace(0, 5, 3), np.linspace(0, 5, 3)])

    def testGoalCost(self):

        # Drive with 1 ms towards goal
        x = np.array([0, 0, 0, 1, 0])
        times, collision_point, idx = dwa.toLineTime(x, self.fmmpath)
        logger.debug("Done")

    def testPlot(self):
        start_angle = np.pi
        v_start = 0
        fig, ax = plt.subplots(figsize=(10, 10))
        x0, x1 = -5, 5
        y0, y1 = -5, 5
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.plot(self.fmmpath[0, :], self.fmmpath[1, :], "-xb")
        nx = 6
        ny = 6
        goal = self.fmmpath
        timefield = np.ones((ny, nx)) * np.nan
        for ii, i in enumerate(np.linspace(x0, x1, nx)):
            for jj, j in enumerate(np.linspace(y0, y1, ny)):
                x = np.array([i, j, start_angle, v_start, 0])
                times, collision_point, idx = dwa.toLineTime(x, goal)

                # ax.text(x[0],x[1],f"{collision_point[0]:6.2f}|{collision_point[0]:6.2f}|{times:6.2f}")
                # if idx != -1 or True:
                #     timefield[jj,ii] = times

                dwa.plot_dw(x, dwa.config, ax)
                u, c = dwa.dwa_control(x, dwa.config, goal)
                xt = [*x[:3], *u]
                dangle = dwa.heading_angle(goal, xt, dwa.config)
                # u,c = dwa.calc_final_input(x,[-0.5,0.5,-0.5,0.5],dwa.config,goal)
                xx, yy = dwa.getPath([*x[:3], *u], dwa.config.predict_time, 10)
                ax.plot(xx[0], yy[0], '+')
                ax.plot(xx, yy, '-')
                ax.text(x[0], x[1], f"{u[0]:6.2f}|{u[0]:6.2f}|{dangle/np.pi*180:6.2f}")

                xx, yy = dwa.getPath(x, dwa.config.predict_time, 10)
                ax.plot(xx, yy, "--k")
                # ax.text(xx[0],yy[0],f"{collision_point[0]:6.2f}|{collision_point[0]:6.2f}")
                # plt.show()

        # ax.imshow(timefield,extent=[-10,10,-10,10],alpha=0.6,origin='lower')
        plt.show()
        logger.debug("")

    def testDwaLineCol(self):
        tic = time.perf_counter()
        d, t = dwa.dwa_line_coll([0, 0, 0, -3, 0, 0.3], [2, -1], [2, 1])
        toc = time.perf_counter() - tic
        logger.debug(f"Took {toc}")
        logger.debug("test")

    def testFilterInputPlot(self):
        from matplotlib import gridspec
        x = [0.25, 0.25, 0.3, 0, 0]
        config = dwa.generate_inter_config()
        u_in = [0.08, 0.1]
        dt = 0.1
        n = 50
        states = np.zeros((n, 5))
        t = np.linspace(0, dt * n, n, False)
        x0, x1, y0, y1 = domain = dwa.interconfig.domain
        for i in range(n):
            dw = dwa.calc_dynamic_window(x, config)
            u_f, cost = dwa.calc_filter_input(x, dw, u_in, config)
            # u_f,cost = dwa.calc_filter_input_cont(x,dw,u_in,config)
            xn = dwa.motion(x, u=u_f, dt=dt)
            x = xn
            states[i, :] = x
        zv = np.zeros(n)
        fig = plt.figure()
        gs = gridspec.GridSpec(4, 5)
        ax = fig.add_subplot(gs[:, :2])
        ax.plot(*dwa.domain_plot_coordinates(domain), '-.k', label='domain')
        ax2 = fig.add_subplot(gs[0, 2:])
        ax3 = fig.add_subplot(gs[1, 2:])
        ax4 = fig.add_subplot(gs[2, 2:])
        ax5 = fig.add_subplot(gs[3, 2:])
        ax.plot(states[:, 0], states[:, 1], label='Trajectory')
        ax2.plot(t, states[:, 0], label='x')
        ax2.plot(t, zv + x0, '-.k', label='bounds')
        ax2.plot(t, zv + x1, '-.k')
        ax3.plot(t, states[:, 1], label='y')
        ax3.plot(t, zv + y0, '-.k', label='bounds')
        ax3.plot(t, zv + y1, '-.k')
        ax4.plot(t, states[:, 3], label='$\\upsilon$')
        ax4.plot(t, states[:, 3] * 0 + u_in[0], '-.k', label='$\\upsilon_{ref}$')
        ax5.plot(t, states[:, 4] * 0 + u_in[1], '-.k', label='$\\dot \\theta_{ref}$')
        ax5.plot(t, states[:, 4], label='$\\dot \\theta$')
        ax.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        ax5.legend()
        ax.set_xlabel("Position [m]")
        ax5.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")
        ax2.set_ylabel("S [m]")
        ax3.set_ylabel("S [m]")
        ax4.set_ylabel("v [m $s^{-1}$]")
        ax5.set_ylabel("$\\omega$ [rad $s^{-1}$]")
        # ax.equal()
        plt.show()

    def testDwaLineColPlot(self):
        "Tests in the trajectories limited by the time before collision indeed stay in the domain.."
        fig, ax = plt.subplots(figsize=(10, 10))
        nx = 11
        ny = 11
        nth = 11
        x0 = y0 = 0
        x1 = y1 = 1.5
        for aa, a, in enumerate(np.linspace(0, 2 * np.pi, nth)):
            for ii, i in enumerate(np.linspace(x0, x1, nx)):
                for jj, j in enumerate(np.linspace(y0, y1, ny)):
                    x = [i, j, a, 0.3, 0.3]
                    d1, t1 = dwa.dwa_line_coll(x, [x0, y0], [x0, y1])
                    d2, t2 = dwa.dwa_line_coll(x, [x0, y1], [x1, y1])
                    d3, t3 = dwa.dwa_line_coll(x, [x1, y1], [x1, y0])
                    d4, t4 = dwa.dwa_line_coll(x, [x1, y0], [x0, y0])
                    t = np.min([t1, t2, t3, t4])
                    if i > 0 and j > 0 and i < 1 and j < 1:
                        xx, yy = dwa.getPath(x, t, 5, linear=False)
                        ax.plot(xx, yy)
        ax.set_xlim([-0.2, 1.2])
        ax.set_ylim([-0.2, 1.2])
        ax.set_xlabel("Position [m]")
        ax.set_ylabel("Position [m]")
        plt.show()
        tic = time.perf_counter()
        toc = time.perf_counter() - tic
        logger.debug(f"Took {toc}")
        logger.debug("test")

    def tearDown(self):
        pass


suite = unittest.TestSuite()

# suite.addTest(TestDwa('testGoalCost'))
# suite.addTest(TestDwa('testPlot'))
# suite.addTest(TestDwa('testDwaLineCol'))
# suite.addTest(TestDwa('testDwaLineColPlot'))
suite.addTest(TestDwa('testFilterInputPlot'))

if __name__ == '__main__':
    # logging.getLogger("vision").setLevel(logging.DEBUG)
    logger.debug("load files")
    unittest.TextTestRunner(verbosity=2).run(suite)
