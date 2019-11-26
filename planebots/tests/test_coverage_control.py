import logging
# logger = logging.getLogger(__name__)
import unittest

import matplotlib.pyplot as plt
import numpy as np
import skfmm
from scipy.spatial import Voronoi, voronoi_plot_2d

from planebots import coverage as cc

logger = logging.getLogger("vision")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
# import planebots
import datetime
from scipy import interpolate


class TestCoverage1(unittest.TestCase):
    points = np.array([[0.75, 0.9], [0.3, 0.8], [0.1, 0.5], [0.7, 0.9], [.5, .1], [1, 0]])
    dim = np.array((11, 11))
    points_idces = np.array(np.random.random((6, 2)) * (dim - 1), np.uint8)
    # points_idces = np.array(points*dim,np.uint8)
    points = np.array(points_idces / (dim - 1))

    Z = np.zeros(dim)
    extent = np.array([-0.5 / dim[0], 1 + 0.5 / dim[0], -0.5 / dim[1], 1 + 0.5 / dim[1]])
    s_1dx = np.linspace(0.0, 1.0, dim[0])
    s_1dy = np.linspace(0.0, 1.0, dim[1])
    # Make the grid for the approximation:
    sxacc, syacc = np.meshgrid(s_1dx, s_1dy)
    qs = np.array([sxacc.ravel(), syacc.ravel()]).T
    sx, sy = sxacc.ravel(), syacc.ravel()

    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger("vision")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        axes = (ax1, ax2, ax3, ax4)

        cls.axes = axes
        cls.fig = fig

    @classmethod
    def tearDownClass(cls):
        for ax in cls.axes:
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlim(-0.1, 1.1)
        datefmt = '%d-%m-%Y %I %M %S'
        svn = datetime.datetime.now().strftime(datefmt)
        plt.savefig(f'test_{svn}.pdf')
        plt.show()
        logger.debug("Ending tests")

    def setUp(self):
        logger.debug(f"Setting up test: {self._testMethodName}")

    def testLloyd(self):

        points, energy, motion = cc.cvt_2d_sampling.init(self.points, 10, 12)
        logger.debug("Points")

    def testProduction(self):
        logger.debug("Test the production")
        productMap = np.zeros(self.dim, np.float32)
        rv = np.zeros(len(self.qs))
        for p in self.points[:1]:
            rvi = cc.alpha_i(p, self.qs)
            rv = rv + rvi

        rvplot = rv / max(rv)
        # ax.imshow(rvplot.reshape(self.dim[1::-1]),extent=[0,1,0,1])
        plt.show()
        # for q in self.qs:
        #     rv = cc.alpha_i(p,q)
        #     logger.debug("Pq")

    def testSpeedFcn(self):
        """Speed function for FMM, defined as F0 + FM
        F0 = 1 - max(-d/C^2 + 2 d/c)
        FM = (1 - e**(-beta* M(p))**-1

        F0 gives a speed boost when not close to the border:
        F0 = 1 on border
        F0 = 1/4 on C/2
        F0 = 0 on C
        F0 = 0 on >>C


        """
        Mk = 1
        beta = 1
        FM = (1 - np.exp(-beta * Mk)) ** -1

    # def testPathCost(self):
    #     for point in route:
    #         pass

    def testFlow(self):
        logger.debug("Creating an initial coverage grid")
        zi = cc.alpha_i([0.51, 0.51], self.qs)
        logger.debug("Filling with random levels")
        random_points = np.random.random((10, 2))
        for pt in random_points:
            zi += cc.alpha_i(pt, self.qs)
        from scipy.ndimage.filters import gaussian_filter
        # Blurring for a 'nice' field
        zi = gaussian_filter(zi.reshape(*self.dim), sigma=2.5).ravel()

        self.axes[0].imshow(zi.reshape(*self.dim), extent=self.extent, origin='lower')
        # Convert to Improvement function:
        res = cc.calc_Mi(self.qs, zi, Zstar=10, alpha_i=cc.alpha_i)
        self.axes[1].imshow(res.reshape(*self.dim), extent=self.extent, origin='lower')

        logger.debug(f"max={max(zi)}, min={min(zi)}")
        logger.debug(f"max={max(res)}, min={min(res)}")

        logger.debug("Generating central voronoi points")
        centralvoronoipoints, energy, motion = cc.cvt_2d_sampling.init(self.points, 5, 30)
        s, g = self.qs, centralvoronoipoints
        k = np.zeros_like(self.sx, np.uint8)
        for idx, ss in enumerate(s):
            k[idx] = np.argmin([np.inner(gg - ss, gg - ss) for gg in g])
        k_rect = k.reshape(*self.dim)
        res_rect = res.reshape(*self.dim)
        res_maxes_point = np.zeros_like(self.points)

        beta = 1
        FM = (1 - np.exp(-beta * res_rect)) ** -1
        speed = FM

        data1 = np.zeros((10, len(self.points)))

        cmap = plt.get_cmap('tab20c')
        # for i in range(len(self.points)):
        for i in range(1):
            # find the index in the potential field:
            point_idx = tuple(self.points_idces[i])
            point_idx = tuple(np.array(centralvoronoipoints[i] * self.dim, np.uint8))
            # find the index of the corresponding voronoi region:
            ki = k_rect[point_idx]
            # finding the max in each region:
            idces = np.where(k_rect == ki)
            idx = np.argmax(res_rect[idces])
            # maxpos_i =np.array(idces)[:,idx]
            maxpos_i = list(zip(*idces))[idx]
            logger.debug(f"{i} {ki} Found a max {res_rect[maxpos_i]} at {maxpos_i}")
            res_maxes_point[ki, :] = [self.sxacc[maxpos_i], self.syacc[maxpos_i]]

            mask = np.where(k != ki, True, False)
            phi = np.ones_like(res_rect, np.int32)

            phi[point_idx] = -1
            # phi_mask = np.ma.MaskedArray(phi,mask)
            logger.debug("Planning path to max..")
            # data1[:6,i]=[i,ki,*maxpos_i,point_idx,mask[maxpos_i]]

            # tuple(self.points[i]*self.dim)

            fmm_times = skfmm.travel_time(phi, speed, dx=[1 / 20, 1 / 20])

            fmm_interp = fmm_times.copy()
            intpfun = interpolate.RectBivariateSpline(self.s_1dx, self.s_1dy, fmm_interp, bbox=[0, 1, 0, 1])

            starting_point = res_maxes_point[ki, :]
            test = intpfun(*starting_point)
            # path = x,y =  cc.gradientDescent(starting_point,intpfun,0.01,0.01)
            # self.axes[3].plot(x,y)
            # self.axes[3].scatter(*starting_point,500,color=cmap.colors[i])
            self.axes[3].scatter(*self.points[i], 200, color=cmap.colors[i])
            self.axes[3].imshow(np.ma.MaskedArray(fmm_times, mask), extent=self.extent, origin='lower')
            # self.axes[3].leg
        self.axes[1].scatter(*centralvoronoipoints.T, 300, "r")

        self.axes[2].imshow(k.reshape(*self.dim), extent=self.extent, origin='lower')
        # self.axes[1].scatter(*centralvoronoipoints.T,300,"r")
        self.axes[1].scatter(*res_maxes_point.T, 300, "g")
        self.axes[2].scatter(*res_maxes_point.T, 300, "g")
        logger.debug("Plot the Voronoi")
        vor = Voronoi(centralvoronoipoints)
        voronoi_plot_2d(vor, ax=self.axes[1])
        voronoi_plot_2d(vor, ax=self.axes[3])

        # Find local minima
        # Find paths
        # DWA

        logger.debug("Points")

    def testGradientDescent(self):
        from scipy import optimize, interpolate
        def field(x, y):
            return (x - 0.5) ** 2 + (y - 0.5) ** 2

        # def field(x,y):
        #     if x >= 0 and x <=1 and y >= 0 and y <= 1:
        #         return x + y*y
        #     else:
        #         return 1e5

        Z = field(self.sxacc, self.syacc)
        intpfun = interpolate.interp2d(self.s_1dx, self.s_1dy, Z)
        logger.debug("Interpolate the coverage field since the path planning is more granular than the field:")
        intpfun = interpolate.RectBivariateSpline(self.s_1dx, self.s_1dy, Z, bbox=[0, 1, 0, 1])
        undef = intpfun(10, 10)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)

        # pts = np.linspace(0,1,20)
        # xg,yg = np.meshgrid(pts,pts)
        # Z = intpfun(xg.ravel(),yg.ravel())

        fminans = optimize.fmin(lambda x: intpfun(x[0], x[1]), [0.9, 0.9],
                                # epsilon=1e-5,
                                full_output=True,
                                retall=True,
                                disp=True)
        # plt.imshow(Z.reshape(20,20),extent=[0,1,0,1],origin='lower')
        # plt.show()
        pts = np.array(fminans[5]).T
        ax.plot(pts[0], pts[1], ':r')

        self.pts = np.zeros((2, 100)) * np.nan
        self.i = 0

        def showstep(X):
            # global pts,i
            self.pts[:, self.i] = X
            self.i += 1
            # Perform a gradient descent on the FMM field to find the trajectory of the agent.
            # Could be made faster by supplying a gradient.

        res = optimize.fmin_tnc(lambda x: intpfun(x[0], x[1]),
                                [0.9, 0.9],
                                epsilon=1e-5,
                                eta=0.1,
                                disp=5,
                                approx_grad=True,
                                callback=showstep,
                                stepmx=0.005)
        p0 = [0.9, 0.9]
        delta = 0.01
        step = 0.01
        p = p0
        self.descent = np.zeros((2, 100)) * np.nan
        for i in range(100):
            dx = delta
            dy = delta
            z1 = intpfun(p[0], p[1])
            dzx = intpfun(p[0] + delta, p[1]) - z1
            dzy = intpfun(p[0], p[1] + delta) - z1
            dzxm = intpfun(p[0] - delta, p[1]) - z1
            dzym = intpfun(p[0], p[1] - delta) - z1
            # We are in a global minimum if all neighbouring points are higher.
            if dzx > 0 and dzy > 0 and dzxm > 0 and dzym > 0:
                break
            function_delta = np.linalg.norm([dzx, dzy]) / delta
            grad = np.array([dzx, dzy]).ravel()
            direction = -grad / np.linalg.norm(grad)
            logger.debug("Done")
            pnew = p + step * direction
            self.descent[:, i] = pnew
            p = pnew
        ax.plot(self.pts[0], self.pts[1], ':k')
        ax.plot(self.descent[0], self.descent[1], ':b')
        plt.show()
        logger.debug("test")

    def test_fmm(self):
        q = self.qs
        Zstar = 5
        Zi = cc.alpha_i([1, 0], q) + cc.alpha_i([0.8, 0.5], q)
        Znorm = -Zi / Zstar + 1

        phi = np.zeros(self.dim) - 1
        phi[5, 5] = 1
        # speed = Znorm.reshape(self.dim)
        speed = np.ones_like(phi)
        ans = skfmm.travel_time(phi, speed, dx=.1)
        ans = skfmm.distance(phi, dx=.1)

        logger.debug("lalala")
        fig, ax = plt.subplots(figsize=(10, 10))

        plt.contour(ans)
        plt.show()

    def test_Mp(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlim(-0.1, 1.1)

        q = self.qs
        Zstar = 5
        Zi = cc.alpha_i([0, 0], q)
        Znorm = -Zi / Zstar + 1
        ax1.imshow(Znorm.reshape(self.dim[1::-1]), extent=[0, 1, 0, 1], origin='lower')
        Mp = np.zeros_like(Znorm)
        for idx, p in enumerate(q):
            a_kqi = cc.alpha_i(p, q)
            num = np.multiply(Znorm, a_kqi)
            den = a_kqi
            Mpi = np.sum(num) / np.sum(den)
            Mp[idx] = Mpi
            # Alternative calculation:
            nonzero = np.where(a_kqi != 0)
            num = np.multiply(Znorm[nonzero], a_kqi[nonzero])
            den = a_kqi[nonzero]
            Mpi = np.sum(num) / np.sum(den)
            Mp[idx] = Mpi

        ax2.imshow(Mp.reshape(self.dim[1::-1]), extent=[0, 1, 0, 1], origin='lower')
        plt.show()

    def test_linecol(self):
        from planebots.control import controllers
        R = 1
        x = np.array([1, 0, np.pi / 2, 2 * R * np.pi, 2 * R * np.pi])  # 1 round per second
        logger.debug("Testing with one collision")
        dist, time = controllers.dwa_line_coll(x, np.array([0, 0]), np.array([0, 2]))
        self.assertEqual(dist, np.pi / 2)
        self.assertEqual(time, 0.25)
        logger.debug("Testing with one collision inversed")
        dist, time = controllers.dwa_line_coll(x, np.array([0, 2]), np.array([0, 0]))
        self.assertEqual(dist, np.pi / 2, "Tested")
        self.assertEqual(time, 0.25)
        logger.debug("Testing with two collisions")
        dist, time = controllers.dwa_line_coll(x, np.array([0, 2]), np.array([0, -2]))
        logger.debug("Testing with no collisions, line inside circle")
        dist, time = controllers.dwa_line_coll(x, np.array([0, 0.5]), np.array([0, -0.25]))
        self.assertEqual(dist, np.inf)
        self.assertEqual(time, np.inf)

        logger.debug("Testing with no collisions, line outside circle")
        dist, time = controllers.dwa_line_coll(x, np.array([10, 10]), np.array([15, 15]))
        self.assertEqual(dist, np.inf)
        self.assertEqual(time, np.inf)

        p1, p2 = controllers.dwa_calc_point_collision(x, np.array([0, 1]), 1, 0)
        p1, p2 = controllers.dwa_calc_point_collision(x, np.array([0, 1]), 1, 0.2)
        p1, p2 = controllers.dwa_calc_point_collision(x, np.array([0, -1]), 0, 0)
        p1, p2 = controllers.dwa_calc_point_collision(x, np.array([0, 1]), 0, 0)
        p1, p2 = controllers.dwa_calc_point_collision(x, np.array([-1, 0]), 0, 0)
        p1, p2 = controllers.dwa_calc_point_collision(x, np.array([0, 1]), 1, 0)

        logger.debug("lalalal")

    def test_DWA(self):
        from planebots.control import dwa
        config = dwa.Config()
        # config.max_speed = 0.6 # [m/s] = max elisa3
        # config.min_speed = -0.6 # [m/s] = min elisa3
        # config.dt = 0.2
        # config.robot_radius = 0.05
        # config.max_yawrate = 2* np.pi
        # config.max_dyawrate = 2 * np.pi /5  # [rad/ss]
        # config.obstacle_cost_gain =1
        # config.to_goal_cost_gain = 0.8
        config.robot_radius = 1
        config.max_accel = 0.5
        ob = np.array([[-1, -1],
                       [0, 2],
                       [4.0, 2.0],
                       [5.0, 4.0],
                       [5.0, 5.0],
                       [5.0, 6.0],
                       [5.0, 9.0],
                       [8.0, 9.0],
                       [7.0, 9.0],
                       [12.0, 12.0]
                       ])
        config.max_speed = 0.6
        config.min_speed = -0.6

        config.v_reso *= 3
        config.yawrate_reso *= 4  # config.predict_time = 0.05
        x = np.array([0.0, 0.0, 9 * np.pi / 8.0, 0.0, 0.0])
        u = np.array([0.0, 0.0])
        logger.debug("Starting DWA")
        goal = np.array([10, 10])
        traj = np.array(x)
        cnt = 0
        while True:
            u, ptraj = dwa.dwa_control(x, u, config, goal, ob)
            x = dwa.motion(x, u, config.dt)  # simulate robot
            cnt += 1

            traj = np.vstack((traj, x))  # store state history
            dist_to_goal = np.linalg.norm(x[:2] - goal[:2])
            logger.debug(
                f"t={config.dt*cnt:5.3f} Distance to goal: {dist_to_goal:4.2f}: ({x[0]:5.3f},{x[1]:5.3f}) speeds:({x[3]:5.3f}ms,{x[4]/np.pi*180:5.3f}deg/s)")
            if dist_to_goal <= config.robot_radius:
                break
        fig, ax = plt.subplots(figsize=(10, 10))
        # ax.set_ylim(0,1)
        # ax.set_xlim(0,1)
        ax.plot(traj[:, 0], traj[:, 1])
        ax.plot(ob[:, 0], ob[:, 1], "ok")
        plt.show()
        logger.debug("Starting DWA")
        from planebots.control import model
        model.Agent


suite = unittest.TestSuite()
# suite.addTest(TestCoverage1('testProduction'))
# suite.addTest(TestCoverage1('test_Mp'))
# suite.addTest(TestCoverage1('test_fmm'))
# suite.addTest(TestCoverage1('testGradientDescent'))
# suite.addTest(TestCoverage1('test_DWA'))
# suite.addTest(TestCoverage1('test_linecol'))
# suite.addTest(TestCoverage1('testLloyd'))
suite.addTest(TestCoverage1('testFlow'))

if __name__ == '__main__':
    # suite = unittest.TestSuite()
    # suite.addTest(TestCoverage1.testProduction)
    # # suite.addTest(TestCoverage1.testLloyd)
    #
    # # unittest.TextTestRunner(suite)
    #
    # points =np.random.random((10,2))

    unittest.TextTestRunner(verbosity=2).run(suite)
