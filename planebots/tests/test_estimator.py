# from vision import coverage_control
import logging
# logger = logging.getLogger(__name__)
import unittest

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("vision")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
# import vision
# import datetime
# from scipy import interpolate
from planebots.control import controllers
import filterpy
from filterpy.kalman import MerweScaledSigmaPoints

dt = 0.1
sigmas = MerweScaledSigmaPoints(5, alpha=.1, beta=2., kappa=1.)

import time


# import vision.model as model

def tick2thomega(ticks, dt):
    thomega = np.array([0.5 * (ticks[0] + ticks[1]), (ticks[0] - ticks[1]) / 0.0408]) / 170  # wheelbase of 40.8mm
    return thomega


ukf_e = controllers.ukf_e
ukf_v = controllers.ukf_v


class TestEstimator(unittest.TestCase):
    x = np.array([0.5, 0, np.pi / 2, 0, 0])
    n = 100
    states = np.zeros((n, 5), np.float)
    states2 = np.zeros((n, 5), np.float)
    states_est = np.zeros((n, 5), np.float)

    Xkm1hatUKF = np.array([0, 0, 0, 0, 0])
    Pkm1EKF = np.eye(5) * 10E-4  # Initial covariance matrix, make heigher with bad estimate
    Pkm1UKF = np.eye(5) * 10E-7
    # UKF tunable parameters:
    alpha = 1e-3  # default, tunable
    ki = 0  # default, tunable
    beta = 2  # default, tunable gaussian white noise

    def testFK(self):
        n = 1000
        states = np.zeros((n, 8))
        for i in range(n - 1):
            j = i + 1
            states[j] = controllers.f_kalman(states[i], u=np.array([1, 2]) * 1, dt=0.1)
        fig, ((ax, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 10))
        ax.plot(states[:, 0] * 1000, states[:, 3] * 1000, '-*', label='kalman_vision')
        ax.legend()
        plt.show()

    def testKbias(self):
        vis = np.load("vision_n4003-71s.npz")
        eli = np.load("elisa_n1923-71s.npz")

    def testBias(self):
        from planebots.control import dwa
        from matplotlib import pyplot as plt
        fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        thv0 = 0
        xstart = [0, 0, thv0, 1, 1]
        times = 2 * np.pi * 5 / 6
        n = 70
        xxv, yyv = dwa.getPath(xstart, times, n, linear=False)
        thv = np.linspace(0, times * xstart[4], n)

        the0 = np.pi / 2
        xstart = [2, 2, the0, 1, 0.9]
        xxe, yye = dwa.getPath(xstart, times, n, linear=False)
        the = np.linspace(0, times * xstart[4], n) + the0

        xeb, yeb = xxe * 0 + xxe[0], yye * 0 + yye[0]
        thb = -the + thv + the * 0
        for i in range(n):
            if not i % 10 or i == 0:
                xeb[i], yeb[i] = xxe[i], yye[i]
                thb[i] = the[i] - thv[i]
            else:
                xeb[i], yeb[i], thb[i] = xeb[i - 1], yeb[i - 1], thb[i - 1]
                xxv[i], yyv[i], thv[i] = xxv[i - 1], yyv[i - 1], thv[i - 1]
        th = thb[0]

        def rm(th):
            rotmat = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            return rotmat

        xxodom = np.zeros_like(xxe)
        yyodom = np.zeros_like(xxe)
        thodom = np.zeros_like(xxe)
        for i in range(n):
            deltax = xxe[i] - xeb[i]
            deltay = yye[i] - yeb[i]
            thd = thb[i]
            x, y = np.matmul(rm(-thd), np.array([deltax, deltay]).T)
            xxodom[i] = x + xxv[i]
            yyodom[i] = y + yyv[i]
            thodom[i] = the[i] - thb[i]

        ax.plot(xxv, yyv, label='xyv')
        ax3.plot(thv, label='thv')
        ax3.plot(the, label='the')
        ax3.plot(thb, label='thb')
        ax4.plot(thodom, label='xyv')
        ax.plot(xxe, yye, label='xye')
        ax2.plot(xxodom, yyodom, '+', label='xye')
        ax.legend()
        plt.show()
        logger.debug("Done")

        def f_km(x, dt):
            xk1 = np.zeros_like(x)
            xk1[:5] = dwa.motion(x[:5])
            return xk1

        dxe = np.diff(xxe)
        dye = np.diff(yye)
        dye = np.diff(yye)

        xkm = np.zeros_like(xxe)
        ykm = np.zeros_like(yye)
        for i in range(1, n):
            pass

    def testWithData(self):
        vis = np.load("vision_n4003-71s.npz")
        eli = np.load("elisa_n1923-71s.npz")
        fig, ((ax, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 10))
        xyth = vis['s_z_xyth']
        v_ts = vis['s_ts']
        ids = vis['s_ids']
        z_ids = vis['s_z_ids']
        idx_elisa = 1
        e_ts = eli['s_z_ts'][:, idx_elisa, :]
        tstart = 10.0
        t0 = e_ts[0]
        idx_start = np.where(e_ts - t0 > tstart)[0][0]
        timespan = min(6.0, e_ts[-2] - t0 - tstart)
        idx_end = np.where(e_ts - e_ts[idx_start] >= timespan)[0][0]
        idx_end = min(idx_end, len(e_ts))
        idx_len = idx_end - idx_start
        v_idx_start = np.where(v_ts - t0 > tstart)[0][0]
        v_idx_end = np.where(v_ts - v_ts[v_idx_start] > timespan)[0][0]
        v_idx_end = min(v_idx_end, len(v_ts))
        v_idx_len = v_idx_end - v_idx_start
        z10ids = z_ids[v_idx_start:v_idx_end, 10]
        z10xyth = xyth[v_idx_start:v_idx_end, 10]
        v_ts = v_ts[v_idx_start:v_idx_end]
        e_ts = e_ts[idx_start:idx_end]
        logger.debug("done")
        e_xyth = eli['s_z_xyth'][idx_start:idx_end, idx_elisa, :]
        e_imu = eli['s_z_imu'][idx_start:idx_end, idx_elisa, :]
        e_u = eli['s_z_u'][idx_start:idx_end, idx_elisa, :]
        e_odom = eli['s_z_odom'][idx_start:idx_end, idx_elisa, :]
        e_ts = eli['s_z_ts'][idx_start:idx_end, idx_elisa, :]
        e_xythodom = np.zeros_like(e_xyth)
        e_xythmodel = np.zeros((idx_len, 8))
        e_xythkalman_e = np.zeros((idx_len, 8))
        e_z = np.zeros((idx_len, 3))
        e_d_imu = np.zeros_like(e_imu)
        e_ticks = np.zeros_like(e_u)

        # Get the velocities from the odometry counter
        ltick = e_odom[:, 0]
        rtick = e_odom[:, 1]
        theta = 0
        x0 = [*z10xyth[0, :2] / 1000, z10xyth[0, 2], 0, 0]

        x_odom = x0
        x_model = [x0[0], 0, 0, x0[1], 0, 0, x0[2], 0]
        ukf_e.x = x_model.copy()

        e_dts = np.zeros_like(e_ts)
        ukf_e.predict(u=np.array([0, 0]) * 0, dt=0.1)  # must always be called before the first predict call
        for i in range(idx_len - 1):
            j = i + 1
            ts = e_ts[j] - e_ts[i]
            e_dts[j] = ts
            dr = ltick[j] - ltick[i]
            dl = rtick[j] - rtick[i]
            e_ticks[j] = [dr / ts / 5, dl / ts / 5]
            e_d_imu[j] = e_d_imu[i] + e_imu[i]
            r, l = e_ticks[i]
            r, l = e_u[i]
            uvo = np.array([0.5 * (l + r), (l - r) / 0.0408]) / 170  # wheelbase of 40.8mm
            xnew_model = controllers.f_kalman(x_model, u=uvo * [1, 1] * 0, dt=ts)
            # ukf_e.predict(u=uvo*0, dt=ts)
            z = np.array([e_ticks[i][0], e_ticks[i][1], e_imu[i][0]])
            e_z[i] = z
            zz = z
            xp = ukf_e.x
            ukf_e.update(z, R=np.diag([1, 1, 1]) * 10)
            p2 = ukf_e.x.copy()
            e_xythkalman_e[j] = ukf_e.x
            if np.linalg.norm(ukf_e.x - e_xythkalman_e[i]) > 1:
                logger.warning("Too high")
            # logger.debug(f"Delta of {dr},{dl} \nf:{xnew_model} \nk:{ukf_e.x}")

            x_model = xm1 = xnew_model
            e_xythmodel[j] = np.array(x_model).squeeze()

        e_xythkalman = np.ones((v_idx_len, 8)) * np.nan
        Xfused = np.ones((idx_len + v_idx_len, 8)) * np.nan  # max steps is all steps from both vision and IMU
        tsfused = np.ones((idx_len + v_idx_len, 2)) * np.nan  # max steps is all steps from both vision and IMU
        ts_san = v_ts[np.where(np.isnan(z10xyth[:, 0]) == False)]
        z10xyth_san = z10xyth[np.where(np.isnan(z10xyth) == False)].reshape(len(ts_san), 3)
        c_time = 0
        t_cur = t0

        x0 = x_current = [z10xyth[0, 0] / 1000, 0, 0, z10xyth[0, 0] / 1000, 0, 0, z10xyth[0, 2], 0]
        vi = 0
        ei = 0
        t0 = t = min(ts_san[vi], e_ts[ei])
        j = 0

        try:
            for i in range(10000):
                vt = ts_san[vi]
                et = e_ts[ei]
                if et < vt:  #
                    # first do a prediction
                    ukf_e.x = x_current
                    ts = et - t
                    r, l = e_u[ei]
                    uvo = np.array([0.5 * (l + r), (l - r) / 0.0408]) / 170  # wheelbase of 40.8mm
                    ukf_e.predict(u=uvo, dt=ts)
                    z = e_z[ei]
                    ukf_e.update(z)
                    ei += 1
                    j += 1
                    Xfused[j] = ukf_e.x

                    x_current = ukf_e.x
                    t = t + ts
                    tsfused[j] = t

                else:
                    ts = vt - t
                    r, l = e_u[ei]
                    ukf_v.x = x_current
                    uvo = np.array([0.5 * (l + r), (l - r) / 0.0408]) / 170  # wheelbase of 40.8mm
                    ukf_v.predict(u=uvo, dt=ts)

                    z = z10xyth_san[vi] / [1000, 1000, 1]
                    ukf_v.update(z, R=np.diag([0.2, 0.2, 5]))
                    vi += 1
                    j += 1
                    Xfused[j] = ukf_v.x
                    x_current = ukf_v.x
                    t = t + ts
                    tsfused[j] = t
        except IndexError as e:
            logger.error(e)
            #

        ukf_v.x = [z10xyth[0, 0] / 1000, 0, 0, z10xyth[0, 0] / 1000, 0, 0, z10xyth[0, 2], 0]
        for i in range(len(ts_san) - 1):
            j = i + 1
            ts = ts_san[j] - ts_san[i]
            tic = time.perf_counter()
            uvo = np.array([ukf_v.x[1] * np.cos(ukf_v.x[6]), ukf_v.x[4] * np.sin(ukf_v.x[6])])
            # uvo = [0,0]
            try:
                # if i == 167:
                #     break
                ukf_v.predict(u=uvo, dt=0.1)
                # pass
            except Exception as e:
                logger.error(e)
            z = z10xyth_san[i] / [1000, 1000, 1]
            kxx = ukf_v.x
            ukf_v.update(z, R=np.diag([0.2, 0.2, 0.1]))
            kxx1 = ukf_v.x
            toc = time.perf_counter()
            logger.debug(toc - tic)
            e_xythkalman[i, :] = ukf_v.x

        # ax.plot(e_xythkalman[:,0]*1000,e_xythkalman[:,3]*1000,'-*',label='kalman_vision')
        ax.plot(e_xythmodel[:, 0] * 1000, e_xythmodel[:, 3] * 1000, label='f_kalman_e')
        # ax.plot(e_xythmodel[1:,0],e_xythmodel[1:,3],'-',label='f_kalman_e')
        ax.plot(e_xythkalman_e[1:, 0] * 1000, e_xythkalman_e[1:, 3] * 1000, 'k', label='kalman_e')
        ax.plot(z10xyth[:, 0], z10xyth[:, 1], label='camera')
        # ax.plot(Xfused[:,0]*1000,Xfused[:,3]*1000,'b',label='fused')
        ax4.plot(v_ts, z10xyth[:, 0], label='camera_x')
        ax4.plot(v_ts, z10xyth[:, 1], label='camera_y')
        # ax2.plot(e_ts,e_xythmodel[:,0]*1000,label='f_kalman_e_x')
        # ax2.plot(e_ts,e_xythmodel[:,0]*1000,label='f_kalman_e_x')
        ax4.plot(e_ts, e_xythmodel[:, 0] * 1000, 'r', label='f_kalman_e_x')
        ax4.plot(e_ts, e_xythmodel[:, 3] * 1000, 'r', label='f_kalman_e_y')
        ax4.plot(e_ts, e_xythkalman_e[:, 0] * 1000, 'k', label='kalman_e_x')
        ax4.plot(e_ts, e_xythkalman_e[:, 3] * 1000, 'k', label='kalman_e_y')
        # ax4.plot(tsfused[:],Xfused[:,0]*1000,'m',label='fusedx')
        # ax4.plot(tsfused[:],Xfused[:,3]*1000,'m',label='fusedy')
        # ax5.plot(e_ts,e_xythkalman_e[:,1],'k',label='kalman_e_dx')
        # ax5.plot(e_ts,e_xythkalman_e[:,4],'r',label='kalman_e_dy')
        ax5.plot(e_ts, e_xythkalman_e[:, 7], 'g', label='kalman_e_dth')
        ax5.plot(np.diff(e_ts, axis=0), label='elisa_ts')
        ax5.plot(np.diff(v_ts, axis=0), label='vision_ts')

        ax6.plot(e_ts, e_imu[:, 0], 'g', label='e_a_perp')
        # ax6.plot(e_ts,e_imu[:,1],'r',label='e_a_il')
        ax6.plot(e_ts, -(e_xythkalman_e[:, 1] ** 2 + e_xythkalman_e[:, 4] ** 2) ** 0.5 * e_xythkalman_e[:, 7] * 20, 'k',
                 label='vau')
        ax2.plot(e_ts, e_u[:, 0], label='u_l')
        ax2.plot(e_ts, e_u[:, 1], label='u_r')
        ax2.plot(e_ts, e_ticks[:, 0], label='ticks_r')
        ax2.plot(e_ts, e_ticks[:, 1], label='ticks_r')
        # ax2.plot(e_ts,e_dts*10e4,label='time')
        ax3.plot(v_ts, z10xyth[:, 2], label='camera')
        ax3.plot(e_ts, e_xythmodel[:, 6], 'r', label='f_kalman_e')
        ax3.plot(e_ts, e_xythkalman_e[:, 6], 'k', label='kalman_e')
        # ax3.plot(v_ts,e_xythkalman_e[:,6],label='kalman_e')
        # ax.plot(e_xythodom[:,0],e_xythodom[:,1],label='odometry')

        # ax2.plot(e_ts,e_d_imu[:,0],label='dIMU inline')
        # ax2.plot(e_ts,e_d_imu[:,1],label='dIMU perp')
        # ax2.plot(e_d_imu[:,1],label='dIMU perp')
        # ddperp = e_xythmodel[:,1]*e_xythmodel[:,6]

        # ax4.plot(e_xythkalman[:,0]*1000,e_xythkalman[:,3]*1000,label='kalman')

        # ax3.plot(e_imu[:,0])
        # ax3.plot(e_imu[:,1])
        # ax4.plot(e_xythodom[:,0],e_xythodom[:,1],label='odometry')
        # ax4.plot(e_xythmodel[:,0],e_xythmodel[:,3],label='model')
        ax3.set_title('angle')
        ax.set_title('xy plot of position in mm')
        ax.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        ax5.legend()
        ax6.legend()
        plt.show()

    def testWithData2(self):
        vis = np.load("vision_n4003-71s.npz")
        eli = np.load("elisa_n1923-71s.npz")
        fig, ((ax, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 10))
        # from planebots.plotting.helpers import plotSerieElisa,plotSerieVision,deltas

        ax3.set_title('angle')
        ax.set_title('xy plot of position in mm')
        ax.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        ax5.legend()
        ax6.legend()
        plt.show()

    def testPaper(self):
        # fig, ax = plt.subplots()
        fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        n = self.n
        self.states = np.zeros((n, 8))
        self.states_est = np.zeros((n, 8))
        dt = 0.02
        u = [1, 2]
        x0 = x = np.zeros((8))
        f = lambda x, dt, u: controllers.f_kalman(x, u, dt)
        xhat = f(x0, dt, u)
        # x dx ddx y dy ddy th omega
        x0 = [2, 0, 0, 0, 0, 0, np.pi / 2, 0]
        # --> 2 0 2 + + 0
        xhat = f(x0, dt, u)
        x0 = [2, 0, 0, 0, 0, 0, np.pi / 4, 0]
        xhat = f(x0, dt, u)
        # logger.debug(f(x))

        for i in range(self.n):
            xn = f(x, dt, u)
            self.states[i, :] = xn
            x = xn
        ax.plot(self.states[:, 0], self.states[:, 3])
        ax2.plot(self.states[:, 1], self.states[:, 4])
        ax3.plot(self.states[:, 2], self.states[:, 5])

        def h_imu(x):
            return [x[2], x[5]]

        sigmas = MerweScaledSigmaPoints(8, alpha=.1, beta=2., kappa=1.)

        ukf = filterpy.kalman.UnscentedKalmanFilter(dim_x=8,
                                                    dim_z=2,
                                                    dt=0.1,
                                                    hx=h_imu,
                                                    fx=f,
                                                    points=sigmas,
                                                    sqrt_fn=None,
                                                    x_mean_fn=None,
                                                    z_mean_fn=None,
                                                    residual_x=None,
                                                    residual_z=None)

        ax.plot(self.states[:, 0], self.states[:, 3])
        ax2.plot(self.states[:, 1], self.states[:, 4])
        ax3.plot(self.states[:, 2], self.states[:, 5])

        ukf.x = np.ones((8))
        ukf.Q = np.eye(8) * .02

        for i in range(self.n):
            tic = time.perf_counter()
            ukf.predict(u=np.array([1, 2]), dt=dt)
            w_std = np.array([2, 2])

            z = h_imu(self.states[i]) + np.random.random((2)).T * w_std - w_std / 2
            ukf.update(z)
            toc = time.perf_counter()
            logger.debug(toc - tic)
            self.states_est[i, :] = ukf.x
            # x = xn

        ax.plot(self.states_est[:, 0], self.states_est[:, 3])
        ax2.plot(self.states_est[:, 1], self.states_est[:, 4])
        ax3.plot(self.states_est[:, 2], self.states_est[:, 5])

        plt.show()
        logger.debug("Done")

    def testPlant(self):
        fig, ax = plt.subplots()
        f = lambda x, dt, u: controllers.dwa_moveinaxes(x, u, dt)

        def h(x):
            return x[:3]

        ukf = filterpy.kalman.UnscentedKalmanFilter(dim_x=5,
                                                    dim_z=3,
                                                    dt=0.1,
                                                    hx=h,
                                                    fx=f,
                                                    points=sigmas,
                                                    sqrt_fn=None,
                                                    x_mean_fn=None,
                                                    z_mean_fn=None,
                                                    residual_x=None,
                                                    residual_z=None)

        ukf.x = np.array([0., 0., 0., 0., 0.])
        ukf.predict(u=np.array([1, 1]), dt=0.2)
        # ukf.update()
        ukf.Q = np.eye(5) * .02
        n = self.n
        dt = 0.02
        u = [1, 2]
        x0 = x = ukf.x = self.x
        f = lambda x, u: controllers.dwa_moveinaxes(x, u, dt)
        for i in range(self.n):
            xn = f(x, u)
            self.states[i, :] = xn
            x = xn
        w = np.array([0.1, 0.1, 0.2, 0, 0]) * 2
        w_std = np.std(w, 0)
        self.noisystates = self.states + np.random.random((5, n)).T * w - w / 2
        for idx in range(n):
            z = self.noisystates[idx, :3]
            ukf.predict(u=np.array([1, 2]), dt=dt)
            ukf.update(z)
            xhat = ukf.x
            self.states_est[idx, :] = xhat

        mp = 4
        dt *= mp
        x = self.x

        # for i in range(self.n//mp):
        #     xn = f(x,u)
        #     self.states2[i,:] = xn
        #     x = xn
        # ax.plot(*self.states2.T[:2])

        # for i in range(self.n):
        #     ukf.x = np.array([0., 0., 0., 0.,0.])
        #     ukf.predict(u=np.array([1,1]),dt=0.2)
        #     ukf.update()

        ax.plot(*self.states.T[:2])
        ax.plot(*self.noisystates.T[:2], '*')
        ax.plot(*self.states_est.T[:2])

        plt.show()
        # logger.info()


suite = unittest.TestSuite()

# suite.addTest(TestEstimator('testBias'))
# suite.addTest(TestEstimator('testPlant'))
# suite.addTest(TestEstimator('testPaper'))
# suite.addTest(TestEstimator('testWithData'))
suite.addTest(TestEstimator('testWithData2'))
# suite.addTest(TestEstimator('testFK'))
# suite.addTest(TestEstimator('testKbias'))
# suite.

if __name__ == '__main__':
    logging.getLogger("vision").setLevel(logging.DEBUG)
    logger.debug("load files")
    unittest.TextTestRunner(verbosity=2).run(suite)
