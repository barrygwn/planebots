import datetime
import logging
import os
import time

import cv2
import inputs
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread
from pyueye import ueye

import planebots
from planebots import coverage as cc
from planebots.control import model, controllers
from planebots.output import gctronic
from planebots.vision import detection, ueye_camera

logger = logging.getLogger(__name__)
agents = [model.Agent(mid=gctronic.elisa_ids[i],
                      states=[0, 0, 0, 0, 0],
                      number=gctronic.elisa_numbers[i],
                      scale=1
                      ) for i in range(len(gctronic.elisa_ids))]


class LogHook(logging.Handler):
    """Hook for logging to the GUI text elements"""
    # log_format = logging.Formatter( '%(asctime)s:%(filename)s:%(levelname)s:%(lineno)s: %(message)s')
    # Format doesnt need time, this is already present in telegram
    log_format = logging.Formatter('%(asctime)s:%(filename)s:%(lineno)s:%(levelname)s:\n %(message)s')

    def __init__(self, qtTextSinkLabel=None, intellij=None, log_format=None):
        logging.Handler.__init__(self)
        if log_format:
            self.log_format = logging.Formatter(log_format)
        self.setFormatter(self.log_format)
        self.qtTextSinkLabel = qtTextSinkLabel
        self.qtTextSinkLabel.setText("")

    def emit(self, record):
        msg = self.format(record)
        self.qtTextSinkLabel.setText(msg)


class Elisa3Output(QtCore.QObject):
    measurement_update = QtCore.pyqtSignal()
    nMaxMeasurements = 10000
    nRobots = len(gctronic.elisa_ids)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QtCore.QBasicTimer()
        self.state = 0
        self.count = 0
        self.initNpz()
        self.storeCount = 0
        self.storeData = False
        self.commOpen = False

    def toggle(self, bool):
        if self.state != bool:
            if bool:
                logger.info(f"Starting communication with {gctronic.elisa_numbers}")
                try:
                    if not hasattr(self, 'comm'):
                        self.comm = gctronic.Elisa3Comm(AddressList=gctronic.elisa_numbers, suppressError=False)
                        self.commOpen = True
                    logger.debug("turning on Communication")
                    Comm = self.comm
                    # self.agents = [model.Agent(mid=gctronic.elisa_ids[i],
                    #                            states=[0, 0, 0, 0, 0],
                    #                            number=gctronic.elisa_numbers[i],
                    #                            scale=1
                    #                            ) for i in range(len(gctronic.elisa_ids))]
                    self.state = True
                    for Address in Comm.AddressList:
                        Comm.setRed(Address, 0)
                        # Show the communication is live:
                        Comm.setGreen(Address, 1)
                    self.setRate(100)
                except OSError as e:
                    logger.error(e)
                    self.state = 0
                    self.commOpen = False
                    return
            else:
                for Address in self.comm.AddressList:
                    self.comm.setRed(Address, 1)
                    # Show the communication is live:
                    self.comm.setGreen(Address, 0)
                    self.comm.setLeftSpeed(Address, 0)
                    self.comm.setRightSpeed(Address, 0)
                time.sleep(0.1)
                self.state = 0
                # self.comm.close()
            # self.state = bool
        else:
            pass

    def start_recording(self, *args, **kwargs):
        self.timer.start(0, self)

    def initNpz(self, suffix=datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")):
        n = self.nMaxMeasurements
        m = self.nRobots
        self.suffix = suffix
        self.s_m_ground = np.ones((n, m, 4), np.float64) * np.nan
        self.s_m_prox = np.ones((n, m, 8), np.float64) * np.nan
        self.s_m_battRaw = np.ones((n, m, 1), np.float64) * np.nan
        self.s_m_number = np.ones((n, m, 1), np.float64) * np.nan
        self.s_m_mid = np.ones((n, m, 1), np.float64) * np.nan
        self.s_m_battPerc = np.ones((n, m, 1), np.float64) * np.nan
        self.s_m_select = np.ones((n, m, 1), np.float64) * np.nan
        self.s_m_odomlr = np.ones((n, m, 2), np.float64) * np.nan
        self.s_m_radioQuality = np.ones((n, m, 1), np.float64) * np.nan
        self.s_m_imu = np.ones((n, m, 3), np.float64) * np.nan
        self.s_z_ts = np.ones((n, m, 1), np.float64) * np.nan
        self.s_z_u = np.ones((n, m, 2), np.float64) * np.nan
        self.s_z_uw = np.ones((n, m, 2), np.float64) * np.nan
        self.s_z_imu = np.ones((n, m, 2), np.float64) * np.nan
        self.s_z_count = np.ones((n, m, 1), np.float64) * np.nan
        self.s_z_odom = np.ones((n, m, 2), np.float64) * np.nan
        self.s_z_xyth = np.ones((n, m, 3), np.float64) * np.nan
        self.s_z_out = np.ones((n, m, 2), np.float64) * np.nan
        self.s_z_ref = np.ones((n, m, 2), np.float64) * np.nan
        self.s_z_xhat = np.ones((n, m, 5), np.float64) * np.nan
        self.storeCount = 0

    def storeTimeStep(self, agent):
        i = self.storeCount = self.storeCount % self.nMaxMeasurements
        j = gctronic.elisa_numbers.index(agent.number)
        self.s_m_ground[i, j, :] = agent.m_ground
        self.s_m_number[i, j, :] = agent.number
        self.s_m_mid[i, j, :] = agent.mid
        self.s_m_prox[i, j, :] = agent.m_prox
        self.s_m_battRaw[i, j, :] = agent.m_battRaw
        self.s_m_battPerc[i, j, :] = agent.m_battPerc
        self.s_m_select[i, j, :] = agent.m_select
        self.s_m_odomlr[i, j, :] = agent.m_odomlr
        self.s_m_radioQuality[i, j, :] = agent.m_radioQuality
        self.s_m_imu[i, j, :] = agent.m_imu
        self.s_z_ts[i, j, :] = agent.z_ts
        self.s_z_u[i, j, :] = agent.z_u
        self.s_z_uw[i, j, :] = agent.z_uw
        self.s_z_imu[i, j, :] = agent.z_imu
        self.s_z_count[i, j, :] = agent.z_count
        self.s_z_odom[i, j, :] = agent.z_odom
        self.s_z_xyth[i, j, :] = agent.z_xyth
        self.s_z_out[i, j, :] = agent.vleft, agent.vright
        self.s_z_ref[i, j, :] = agent.lout, agent.rout
        self.s_z_xhat[i, j, :] = agent.x_hat

    def saveTimeSteps(self):
        suffix = self.suffix
        savedir = os.path.join(planebots.packdir, 'data', 'dumps', suffix)
        n = self.storeCount
        if n:
            # if n:
            savename = os.path.join(savedir, f"meas_{suffix}_elisa_n{n}-{int(self.s_z_ts[n-1,0]-self.s_z_ts[0,0])}s")
            try:
                saveArrays = {'s_m_ground': self.s_m_ground[:n, :, :],
                              's_m_number': self.s_m_number[:n, :, :],
                              's_m_mid': self.s_m_mid[:n, :, :],
                              's_m_prox': self.s_m_prox[:n, :, :],
                              's_m_battRaw': self.s_m_battRaw[:n, :, :],
                              's_m_battPerc': self.s_m_battPerc[:n, :, :],
                              's_m_select': self.s_m_select[:n, :, :],
                              's_m_odomlr': self.s_m_odomlr[:n, :, :],
                              's_m_radioQuality': self.s_m_radioQuality[:n, :, :],
                              's_m_imu': self.s_m_imu[:n, :, :],
                              's_z_ts': self.s_z_ts[:n, :, :],
                              's_z_u': self.s_z_u[:n, :, :],
                              's_z_uw': self.s_z_uw[:n, :, :],
                              's_z_imu': self.s_z_imu[:n, :, :],
                              's_z_count': self.s_z_count[:n, :, :],
                              's_z_odom': self.s_z_odom[:n, :, :],
                              's_z_xyth': self.s_z_xyth[:n, :, :],
                              's_z_out': self.s_z_out[:n, :, :],
                              's_z_ref': self.s_z_ref[:n, :, :],
                              's_z_xhat': self.s_z_xhat[:n, :, :]}
                np.savez_compressed(savename, **saveArrays)

                logger.info(f"{n}/{self.nMaxMeasurements} Measurement saved as '{savename}'")
            except Exception as e:
                logger.warning(e)
        else:
            logger.warning("No data available for Elisa output!")

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        if not self.state:
            self.timer.stop()
            logger.warning("Elisa timer stopped due to off state")
            return

        self.count += 1

        for agent in agents:
            if agent.mid in gctronic.elisa_ids and self.commOpen:
                agent.getOdometry(self.comm)
                self.comm.setLeftSpeed(agent.number, agent.lout)
                self.comm.setRightSpeed(agent.number, agent.rout)
                # agent.z_count +=1
                # logger.debug("test")
            self.storeTimeStep(agent)
        self.storeCount += 1
        self.measurement_update.emit()

    def stop_recording(self):
        self.timer.stop()
        self.count = 0

    def setRate(self, valuehz):
        self.timer.stop()
        self.rate = valuehz
        timeoutms = int(valuehz ** -1 * 1000)
        logger.debug(f"Timer for communication with Elisa set every {timeoutms} ms")
        self.timer.start(timeoutms, self)


# class testElisaOut(QtCore.QObject):
#     def __init__(self, rateMs, parent=None):
#         super().__init__(parent)
#         self.timer = QtCore.QBasicTimer()
#
#
#
#     """QtCor."""
#
#     def __init__(self, timeMS, parent=None):
#         super().__init__(parent)
#         self.timer = QtCore.QBasicTimer()
#         self.state = 0
#         self.count = 0
#         self.states = controllers.gamepad_states_default.copy()
#         # self.states["n_agents"] = len(planebots.n)
#
#     def toggle(self, bool):
#         if self.state != bool:
#             if bool:
#                 logger.debug("Starting timer for GamePad")
#                 self.start_updates()
#             else:
#                 self.stop_updates()
#             self.state = bool
#
#     def start_updates(self, *args, **kwargs):
#         self.timer.start(0, self)
#
#     def timerEvent(self, event):
#         if (event.timerId() != self.timer.timerId()):
#             return
#         try:
#             states = controllers.monitor_gamepad(self.states)
#             self.states.update(states)
#             self.count += 1
#         except inputs.UnpluggedError as e:
#             logger.error(e)
#             self.state = 0
#             logger.debug("Redetecting devices...")
#             inputs.DeviceManager()
#             self.stop_updates()
#
#     def stop_updates(self):
#         self.timer.stop()
#         self.count = 0


class GamePad(QtCore.QObject):
    """QtCor."""

    def __init__(self, source=0, parent=None):
        super().__init__(parent)
        self.timer = QtCore.QBasicTimer()
        self.state = 0
        self.count = 0
        self.states = controllers.gamepad_states_default.copy()
        # self.states["n_agents"] = len(planebots.n)

    def toggle(self, bool):
        if self.state != bool:
            if bool:
                logger.debug("Starting timer for GamePad")
                self.start_updates()
            else:
                self.stop_updates()
            self.state = bool

    def start_updates(self, *args, **kwargs):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return
        try:
            states = controllers.monitor_gamepad(self.states)
            self.states.update(states)
            self.count += 1
        except inputs.UnpluggedError as e:
            logger.error(e)
            self.state = 0
            logger.debug("Redetecting devices...")
            inputs.DeviceManager()
            self.stop_updates()

    def stop_updates(self):
        self.timer.stop()
        self.count = 0


class UeyeThread(QThread):
    """Thread to be run independently to fetch camera stills, due to blocking behaviour"""
    signal = QtCore.pyqtSignal('PyQt_PyObject')
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        QThread.__init__(self)
        self.state = 0
        self.ueyeretriever = UeyeRetriever()

    def __del__(self):
        self.wait()

    def run(self):
        self.count = 0
        self.state = 0
        self.state = 1
        nret = self.ueyeretriever.initialize()
        if nret != -1:
            self.frame = np.zeros((ueye_camera.width, ueye_camera.height), np.uint8)
            while True:
                tic = time.perf_counter()
                try:
                    ueye_camera.cameraNewFrame(self.frame, self.ueyeretriever.hCam)
                    self.count += 1
                    # detection.addTextlines(self.frame,[f"{self.count}"])
                    # Emit a signal with the frame:
                    toc = time.perf_counter() - tic
                    self.image_data.emit(self.frame)
                except Exception as e:
                    logger.error(e)
            self.state = 0
        else:
            logger.error("Could not initialize camera!")
            self.state = 0

    def toggle(self, bool):
        if bool != self.state:
            if bool:
                self.start()
            else:
                self.exit()
                self.state = 0


#
# class OdometryThread(QThread):
#     """Thread to be run independently to fetch camera stills, due to blocking behaviour"""
#     signal = QtCore.pyqtSignal('PyQt_PyObject')
#
#     def __init__(self):
#         QThread.__init__(self)
#         self.state = 0
#         self.timer = QtCore.QBasicTimer()
#         # self.comm = gctronic.Elisa3Comm(AddressList=gctronic.elisa_numbers)
#
#     def __del__(self):
#         self.wait()
#
#     def run(self):
#         self.count = 0
#         self.state = 0
#         self.state = 1
#         self.timer.start(0, self)
#     def timerEvent(self, event):
#         if (event.timerId() != self.timer.timerId()):
#             return
#         logger.info("Timer fired")
#         if not self.state:
#             self.timer.stop()
#             logger.warning("Elisa timer stopped due to off state")
#             return
#
#         self.count += 1
#         for agent in agents:
#             logger.debug("Test")
#
#     def toggle(self, bool):
#         if bool != self.state:
#             if bool:
#                 self.start()
#                 logger.info("Starting odometry measurements")
#             else:
#                 logger.info("Stopping odometry measurements")
#                 self.timer.stop()
#                 # self.comm.close()
#                 self.exit()
#                 self.state = 0


class GamePadThread(QThread):
    """Thread to be run independently to fetch gamepad presses, due to blocking behaviour"""
    signal = QtCore.pyqtSignal('PyQt_PyObject')

    def __init__(self):
        QThread.__init__(self)
        self.state = 0

    def __del__(self):
        self.wait()

    def run(self):
        self.count = 0
        self.state = 0
        self.state = 1
        self.gamePad = GamePad()
        self.gamePad.states["n_agents"] = len(gctronic.elisa_numbers)
        self.gamePad.states["current_id"] = 0
        self.gamePad.states["previous_id"] = 1
        while True:
            try:
                new_states = controllers.monitor_gamepad(self.gamePad.states)
                self.gamePad.states.update(new_states)
                logger.debug("Detected a gamepad action...")
                self.count += 1
                self.gamePad.states.update({'count': self.count})
                self.signal.emit(self.gamePad.states)
            except inputs.UnpluggedError as e:
                logger.error(e)
                self.state = 0
                logger.debug("Redetecting devices...")
                inputs.DeviceManager()
                break
        self.state = 0

    def toggle(self, bool):
        if bool != self.state:
            if bool:
                self.start()
            else:
                self.exit()
                self.state = 0


class UsbCamera(QtCore.QObject):
    """Retrieves frames from a usb camera or from a file, if source is an integer, it tries to open a usb device, otherwise
     if source is a string, it parses it as a file path and tries to open it as a video."""
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, source=0, parent=None):
        super().__init__(parent)
        self.cameraport = source
        self.camera = cv2.VideoCapture(self.cameraport)
        self.timer = QtCore.QBasicTimer()
        self.state = 0
        self.count = 0

    def toggle(self, bool):
        if self.state != bool:
            if bool:
                if not self.camera.isOpened():
                    self.camera = cv2.VideoCapture(self.cameraport)
                self.start_recording()
            else:
                self.stop_recording()
            self.state = bool

    def start_recording(self, *args, **kwargs):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        read, data = self.camera.read()
        self.count += 1
        detection.addTextlines(data, [f"{self.count}"])
        if read:
            self.image_data.emit(data)

    def stop_recording(self):
        self.timer.stop()
        self.count = 0
        self.camera.release()


class videoFileReader(QtCore.QObject):
    """Retrieves frames from a usb camera or from a file, if source is an integer, it tries to open a usb device, otherwise
     if source is a string, it parses it as a file path and tries to open it as a video."""
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.source = cv2.VideoCapture(filepath)
        self.timer = QtCore.QBasicTimer()
        self.state = 0
        self.count = 0
        self.filepath = filepath

    def toggle(self, bool):
        if self.state != bool:
            if bool:
                if not self.source.isOpened():
                    self.source = cv2.VideoCapture(self.filepath)
                self.start_recording()
            else:
                self.stop_recording()
            self.state = bool

    def start_recording(self, *args, **kwargs):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return
        try:
            read, data = self.source.read()
        except Exception as e:
            logger.error(e)
        if not read:
            logger.error("Read error")
            self.source.release()
            self.state = False
            self.toggle(True)
            return
        self.count += 1
        detection.addTextlines(data, [f"{self.count}"])
        if read:
            self.image_data.emit(data)

    def stop_recording(self):
        self.timer.stop()
        self.count = 0
        self.source.release()


class DetectionHelper(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)
    detectionPerformed = QtCore.pyqtSignal()
    pnpTried = QtCore.pyqtSignal()
    storeGuiFrames = QtCore.pyqtSignal(str)
    # detectionPerformed = QtCore.pyqtSignal()
    nMaxMarkers = 20
    nMaxMeasurement = 10000

    def __init__(self, parent=None):
        super().__init__(parent)
        logger.debug("Initializing detector")
        self.storeTimeStepInit(self.nMaxMeasurement, storeFrames=False)
        self.agents = model.Agent.fromConfig()
        self.elisaList = model.Agent.fromConfig()
        self.timer = QtCore.QBasicTimer()
        self.state = 0
        self.count = 0
        self.maxRows = 20000
        self.ms = 0
        # default values of the stills
        self.rvec = np.array([[-0.12093959], [-0.12143663], [-0.22531726]])
        self.tvec = np.array([[-158.23656287], [-150.3255944], [648.85777562]])
        self.agentlist = []
        self.np_data = np.zeros((100, 100), np.uint8)
        # Set if the frame is analyzed yet
        self.analyzed = False
        self.analyzeTime = 0
        self.timeStamp = datetime.datetime.now()
        self.calList = []
        self.unknownList = []
        self.aList = []
        self.pnpSuccess = False
        self.continuousPnP = False
        # ts, rvec, tvec
        self.pnpValues = np.array((7, 20000))
        self.pnpStoreCount = 0
        self.storeCount = 0
        # The last detected markers:
        self.rawCorners = []
        self.rawIds = []
        self.rawRej = []
        self.storeData = False

    def storeTimeStepInit(self, n, storeFrames, suffix=datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")):
        self.s_ts = np.ones((n, 1), np.float64) * np.nan
        self.s_tvec = np.ones((n, 3, 1), np.float64) * np.nan
        self.s_rvec = np.ones((n, 3, 1), np.float64) * np.nan
        self.s_corners = np.ones((n, self.nMaxMarkers, 4, 2), np.float64) * np.nan
        self.s_ids = np.ones((n, self.nMaxMarkers), np.float64) * np.nan
        self.s_z_ids = np.ones((n, self.nMaxMarkers), np.uint8) * np.nan
        self.s_z_xyth = np.ones((n, self.nMaxMarkers, 3), np.float64) * np.nan
        self.s_z_xhat = np.ones((n, self.nMaxMarkers, 5), np.float64) * np.nan
        self.s_zmean = np.ones(n, np.float64)
        self.s_zstd = np.ones(n, np.float64)

        # self.s_guiframes = np.ones((n, *self.np_data.size, 3), np.uint8)
        # self.s_rawframes = np.ones((n, *self.np_data.size, 3), np.uint8)
        self.storeCount = 0
        self.storeFrames = storeFrames
        self.suffix = suffix
        self.savedir = os.path.join(planebots.packdir, "data", "dumps", self.suffix)
        self.n = n

        self.measurementRunning = False
        if storeFrames:
            self.makefolder()

    def makefolder(self):
        try:

            if not os.path.exists(self.savedir):
                os.mkdir(self.savedir)
                os.mkdir(os.path.join(self.savedir, 'raw'))
                os.mkdir(os.path.join(self.savedir, 'gui'))
        except Exception as e:
            logger.error(e)

    def storeTimeStep(self, ts, rvec, tvec, corners, ids, z_ids, z_xyth):
        tic = time.perf_counter()
        n = self.storeCount % len(self.s_ts)
        self.s_ts[n, :] = ts
        self.s_tvec[n, :] = tvec
        self.s_rvec[n, :] = rvec
        if len(self.corners):
            self.s_corners[n, :len(corners), :] = corners
            self.s_ids[n, :len(corners)] = ids.ravel()
        self.s_z_ids[n, :] = z_ids
        self.s_z_xyth[n, :] = z_xyth
        self.s_zmean[n] = np.sum(cc.Z) / cc.Z.size
        self.s_zstd[n] = np.std(cc.Z)

        self.storeCount += 1
        t = time.perf_counter() - tic
        if self.storeCount == self.nMaxMeasurement:
            logger.warning("Rolling over visual odometry data!")
            if self.storeData:
                self.saveTimeSteps()
        if self.storeFrames and self.measurementRunning:
            rawsavename = os.path.join(self.savedir, 'raw', f"frame_raw_{self.storeCount:05d}.png")
            guisavename = os.path.join(self.savedir, 'gui', f"frame_gui_{self.storeCount:05d}.png")
            cv2.imwrite(rawsavename, self.np_data)
            self.storeGuiFrames.emit(guisavename)

    def saveTimeSteps(self):

        suffix = self.suffix
        n = self.storeCount
        savename = os.path.join(self.savedir, f"meas_{suffix}_planebots_n{n}-{int(self.s_ts[n-1]-self.s_ts[0])}s")
        saveArrays = {'s_ts': self.s_ts[:n],
                      's_tvec': self.s_tvec[:n],
                      's_rvec': self.s_rvec[:n],
                      's_corners': self.s_corners[:n],
                      's_ids': self.s_ids[:n],
                      's_z_ids': self.s_z_ids[:n],
                      's_zmean': self.s_zmean[:n],
                      's_zstd': self.s_zstd[:n],
                      's_z_xyth': self.s_z_xyth[:n],
                      's_z_xhat': self.s_z_xhat[:n]}

        np.savez_compressed(savename, **saveArrays)
        # savename = os.path.join(self.savedir, f"meas_{suffix}_rawframes_n{n}-{int(self.s_ts[n-1]-self.s_ts[0])}s")

        logger.info(f"{n}/{self.nMaxMeasurement} Measurement saved as '{savename}'")

    def storePnP(self, ts, rvec, tvec):
        self.pnpValues[:, self.pnpStoreCount] = np.array([ts, *rvec, *tvec], np.float32)
        self.pnpStoreCount = self.pnpStoreCount % self.maxRows

    def tryPnP(self):
        try:
            self.pnp(self.np_data)
            self.pnpTried.emit()
        except Exception as e:
            logger.debug(e)

    def pnp(self, np_data, *args, **kwargs):
        tic = time.perf_counter()
        framesize = np_data.shape[1::-1]
        mtxrs, roi = cv2.getOptimalNewCameraMatrix(detection.mtx, 0, (1280, 1024), 1, framesize)
        self.mtx = mtxrs
        # corners, ids, rej = cv2.aruco.detectMarkers(np_data,
        #                                             planebots.MarkerDictionary,
        #                                             None,
        #                                             None,
        #                                             planebots.DetectionParameters,
        #                                             None)

        try:
            pnpSuccess, rvecfound, tvecfound = detection.findPnP(mtxrs,
                                                                 detection.dist,
                                                                 self.rawCorners,
                                                                 self.rawIds,
                                                                 detection.calibration_marker_ids,
                                                                 detection.calibration_markers_mm)
            self.pnpTimeStamp = time.perf_counter()

            logger.debug("Performed pnp: {pnpSuccess}")


        except detection.NoMarkersException as e:
            logger.error(e)
            pnpSuccess = False

        if pnpSuccess:
            rvec, tvec = self.rvec, self.tvec = rvecfound, tvecfound
            # self.storePnP(self.pnpTimeStamp,rvec,tvec)
        toc = time.perf_counter()
        elapsed = tic - toc
        self.pnp_elapsed = elapsed
        self.pnpSuccess = pnpSuccess

    def setDetectFrame_slot(self, np_data):
        """Set the detect frame"""
        self.np_data = np_data
        self.analyzed = False
        # Store the camera matrices corresponding with the frame size:
        self.mtxWindow, self.newMtxWindow = detection.getCameraMatrices(np_data)

    def setRate(self, valuehz):
        self.timer.stop()
        self.rate = valuehz
        timeoutms = valuehz ** -1 * 1000
        logger.debug(f"timer to fire every {timeoutms} ms")
        self.timer.start(timeoutms, self)

    def step(self):
        """Performs a detection step"""
        tic = time.perf_counter()
        if self.continuousPnP:
            self.tryPnP()

        npsz = self.np_data.shape[1::-1]
        mtxdetect, roi = cv2.getOptimalNewCameraMatrix(detection.mtx, 0, detection.original_size, 1, npsz)

        corners, ids, rej = cv2.aruco.detectMarkers(self.np_data,
                                                    planebots.MarkerDictionary,
                                                    None,
                                                    None,
                                                    planebots.DetectionParameters,
                                                    None)
        self.rawCorners, self.rawIds, self.rawRej = corners, ids, rej
        self.corners = corners
        self.ids = ids
        cornersInPlane = list(map(lambda x: detection.correspondence2d3d(x,
                                                                         mtxdetect,
                                                                         detection.dist,
                                                                         self.rvec,
                                                                         self.tvec,
                                                                         detection.calibration_markers_mm), corners))
        self.corners3D = cornersInPlane

        centersInPlane, anglesInPlane = detection.centerMarkers(cornersInPlane)
        detectionObjects = []
        if len(centersInPlane):
            for idx, i in enumerate(ids.ravel()):
                states = [*centersInPlane[idx].ravel()[:2], *anglesInPlane[idx].ravel()[:], 0, 0]
                states = list(map(lambda x: float(x), states))
                agent = model.Agent(mid=i, states=states, number=0)
                detectionObjects.append(agent)
        self.calList = [ag for ag in detectionObjects if ag.mid in detection.calibration_marker_ids]
        self.unknownList = [ag for ag in detectionObjects if
                            ag.mid not in detection.calibration_marker_ids and ag.mid not in detection.elisa_ids]
        # self.elisaList = [ag for ag in detectionObjects if ag.mid in detection.elisa_ids]
        self.elisaList = model.Agent.updateAgents(self.elisaList, centersInPlane, anglesInPlane, ids)

        self.ts = time.perf_counter()
        self.z_ids = np.ones((self.nMaxMarkers)) * np.nan
        self.z_xyth = np.ones((self.nMaxMarkers, 3)) * np.nan

        idsFlat = np.array(ids).ravel()
        for i in range(len(corners)):
            logger.debug("Store in convenient format")
            try:
                mIdx = idsFlat[i]
                xyth = np.array([*centersInPlane[i, 0, :2], anglesInPlane[i]])
                self.z_ids[mIdx] = mIdx
                self.z_xyth[mIdx] = xyth
            except Exception as e:
                logger.warning(e)
                logger.warning(f"Marker id f{mIdx} can not be saved")

        self.storeTimeStep(self.ts,
                           self.rvec,
                           self.tvec,
                           self.corners,
                           self.ids,
                           self.z_ids,
                           self.z_xyth)

        self.aList = [ag for ag in detectionObjects if ag.mid not in detection.calibration_marker_ids]
        agentlist = model.Agent.updateAgents(self.agents, centersInPlane, anglesInPlane, ids)

        self.agentlist = agentlist
        logger.debug("Detection step done")
        self.count += 1
        self.analyzed = True
        toc = time.perf_counter()
        self.analyzeTime = toc - tic
        self.timeStamp = datetime.datetime.now()

        self.detectionPerformed.emit()

    def toggle(self, bool):
        # toggle detection state
        if self.state != bool:
            if bool:
                # We want to switch ON but are OFF:
                self.start_detection()
            else:
                self.stop_detection()
            self.state = bool

    def togglePnP(self, bool):
        # toggle detection state

        self.continuousPnP = bool
        logger.info("Switch pnp Mode")
        self.tryPnP()

    def start_detection(self, *args, **kwargs):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        self.step()
        self.detectionPerformed.emit()

    def stop_detection(self):
        self.timer.stop()
        self.count = 0
        # self.camera.release()


class UeyeRetriever(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        # nRet, hCam = ueye_camera.cameraInit(load_parameters_EEPROM=False)
        # self.nRet = nRet
        # self.hCam = hCam
        self.state = 0
        self.timer = QtCore.QBasicTimer()
        self.frame = np.zeros((ueye_camera.width, ueye_camera.height), np.uint8)
        self.image_data.emit(self.frame)
        self.count = 0
        self.initialized = 0

        self.singleTimer = QtCore.QTimer()

    def initialize(self):
        self.nRet, self.hCam = ueye_camera.cameraInit(load_parameters_EEPROM=False)
        self.setPixelClock(71)
        self.setExposure(3)
        self.setFPS(50)
        return self.nRet

    def toggle(self, bool):
        if self.state != bool:
            if bool:
                # We want to be switched ON but are OFF:
                if not self.initialized:
                    # self.singleTimer.singleShot(1000,)
                    self.initialize()
                if self.nRet < 0:
                    detection.addTextlines(self.frame, ["Camera not connected!"])
                    self.image_data.emit(self.frame)
                    self.state = bool
                else:
                    self.state = 1
                    self.width = ueye_camera.rect_aoi.s32Width.value
                    self.height = ueye_camera.rect_aoi.s32Height.value
                    self.frame = np.zeros((self.height, self.width), np.uint8)
                    self.initialized = True
                    self.start_recording()
            else:
                self.stop_recording()
                self.state = 0
            # else:
            #     self.stop_recording()
        self.state = bool

    def delayedCameraInit(self):
        pass

    def start_recording(self, *args, **kwargs):
        self.timer.start(0, self)

    def camParLoad(self):
        """Load Ueye Camera Parameters from EEPROM"""
        logger.debug("Loading parameters from EEPROM")
        logger.debug("Disabling framegetter:")
        self.stop_recording()
        nullint = ueye._value_cast(0, ueye.ctypes.c_uint)
        if self.state:
            rv = ueye.is_ExitCamera(self.hCam)
            nret, hCam = ueye_camera.cameraInit(True)

            # rvv = ueye.is_ParameterSet(self.hCam,ueye.IS_PARAMETERSET_CMD_LOAD_EEPROM,nullint,nullint)
            # rectAOI = rect_aoi = ueye.IS_RECT()
            # # Can be used to set the size and position of an "area of interest"(AOI) within an image
            # nRet = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))

            self.width = ueye_camera.rect_aoi.s32Width.value
            self.height = ueye_camera.rect_aoi.s32Height.value
            if self.width > 0:
                self.frame = np.zeros((self.height, self.width), np.uint8)
                self.start_recording()
            else:
                detection.addTextlines(self.frame, ["Camera not connected!", "Cannot load parameters"])
                self.image_data.emit(self.frame)
        # while self.state:

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return
        # refresh frame buffer
        ueye_camera.cameraNewFrame(self.frame, self.hCam)
        self.count += 1
        # detection.addTextlines(self.frame,[f"{self.count}"])
        # Emit a signal with the frame:
        self.image_data.emit(self.frame)

    def setPixelClock(self, mhz):
        px_old = ueye.c_uint(0)
        ueye.is_PixelClock(self.hCam, ueye.IS_PIXELCLOCK_CMD_GET, px_old, ueye.sizeof(px_old))
        px = ueye.c_uint(mhz)
        rv = ueye.is_PixelClock(self.hCam, ueye.IS_PIXELCLOCK_CMD_SET, px, ueye.sizeof(px))

    def setExposure(self, ms):
        ms_old = ueye.c_double(0)
        rv = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, ms_old, ueye.sizeof(ms_old))
        ems = ueye.c_double(ms)
        rv = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ems, ueye.sizeof(ems))
        ms_old = ueye.c_double(0)
        rv = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, ms_old, ueye.sizeof(ms_old))
        return rv

    def setFPS(self, fps):
        ms_new = ueye.c_double(fps)
        new = ueye.c_double(0)
        rv = ueye.is_SetFrameRate(self.hCam, ms_new, new)
        return new

    def stop_recording(self):
        self.timer.stop()
        self.count = 0

    # def timerEvent(self, event):
    #     if (event.timerId() != self.timer.timerId()):
    #         return
    #
    #     read, data = self.camera.read()
    #     if read:
    #         self.image_data.emit(data)
    #
    # def stop_recording(self):
    #     self.timer.stop()
    #     self.camera.release()


class imgDisplayWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)
        self.total_count = 0
        self.source_count = 0
        self.width = 640
        self.height = 512
        self.showFrame = np.zeros((self.height, self.width, 3), np.uint8)
        self.np_data = np.zeros((self.height, self.width, 3), np.uint8)

        self.updateDistortionCoeffs()

    def getNewMtx(self):
        framesize = (self.width, self.height)
        mtxwindow, roi = cv2.getOptimalNewCameraMatrix(detection.mtx, 0, detection.original_size, 1,
                                                       (self.width, self.height))
        npsz = self.np_data.shape[1::-1]
        mtxdetect, roi = cv2.getOptimalNewCameraMatrix(detection.mtx, 0, detection.original_size, 1, npsz)
        # recalculate mtx for displaying all image pixels in the ud frame:
        newmtxwindow, roi = cv2.getOptimalNewCameraMatrix(mtxwindow, detection.dist, framesize, 1, framesize)
        newmtxdetect, roi = cv2.getOptimalNewCameraMatrix(mtxdetect, detection.dist, npsz, 1, npsz)

        return mtxdetect, newmtxdetect, mtxwindow, newmtxwindow

    def updateDistortionCoeffs(self):
        framesize = (self.width, self.height)
        mtxrs, roi = cv2.getOptimalNewCameraMatrix(detection.mtx, 0, (1280, 1024), 1, (self.width, self.height))
        # recalculate mtx for displaying all image pixels in the ud frame:
        newmtx, roi = cv2.getOptimalNewCameraMatrix(mtxrs, detection.dist, framesize, 1, framesize)

        self.newmtx = newmtx
        self.mtxrs = mtxrs

    def updateShowFrame(self, image_data):
        self.showFrame = image_data

    # def updateImageData_slot(self, image_data):
    #     """Converts a np array into a suitable format, and displays it"""
    #     self.np_data = image_data

    def showProcessedData_slot(self, np_data):
        """Shows the processed frame on screen"""
        self.showFrame = np_data
        self.image = self.get_qimage(self.showFrame)
        if self.image.size() != self.size():
            sz = self.image.size()
            self.setFixedSize(sz)
        self.update()

    def image_data_slot(self, image_data):
        """Converts a np array into a suitable format, and saves it"""
        self.np_data = image_data
        # self.image = self.get_qimage(image_data)
        # if self.image.size() != self.size():
        #     sz = self.image.size()
        #     self.setFixedSize(sz)

    def dump(self):
        detection.saveFrame(self.showFrame, prefix="dump")

    def get_qimage(self, image: np.ndarray):
        if len(image.shape) == 2:
            proc_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        elif len(image.shape) == 3:
            proc_image = image.copy()
        rsz = cv2.resize(proc_image, (self.width, self.height))
        height, width, colors = rsz.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage
        image = QImage(rsz.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)
        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        # self.image = QtGui.QImage()


class NpLoader(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)
    rawDataSignal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)

    def onEvent(self, event):

        image_data = cv2.cvtColor(detection.stillgray, cv2.COLOR_GRAY2BGR)
        self.image_data.emit(image_data)

    def loadStillAgent(self, bool):
        # Load an image containing calibration markers and agent markers
        if bool:
            image_data = cv2.cvtColor(detection.stillagent, cv2.COLOR_GRAY2BGR)
            self.image_data.emit(image_data)
            self.rawDataSignal.emit(image_data)

    def loadStill(self, bool):
        if bool:
            image_data = cv2.cvtColor(detection.stillgray, cv2.COLOR_GRAY2BGR)
            self.image_data.emit(image_data)
            self.rawDataSignal.emit(image_data)
