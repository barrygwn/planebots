# -*- coding: utf-8 -*-
import datetime
import logging
import os
import sys
from importlib import reload  # Python 3.4+ only.

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets

from planebots.control import control_mapping
from planebots.gui import overlays
from planebots.gui.qt_classes import imgDisplayWidget, NpLoader, \
    UsbCamera, DetectionHelper, Elisa3Output, LogHook, GamePadThread, UeyeThread
from planebots.gui.ui import Ui_MainWindow
from planebots.output import gctronic
from planebots.vision import detection

logging.getLogger("planebots")
cnt = 1
from planebots.control import dwa
from planebots import coverage as cc
from planebots.gui import custom
from planebots.control.observers import ukFilter, h_vision, remap_angle
import time
from planebots.gui import qt_classes

mousepos3d = np.array([30., 30, 0])


class Ui_MainWindow_wc(Ui_MainWindow):
    def __init__(self, MainWindow):
        # initialize the UI retrieved from the QT Designer:
        super(Ui_MainWindow, self).__init__()
        # self.setupUi()
        self.agents = qt_classes.agents
        self.cnt = 1  # Init counter
        self.imShow = imgDisplayWidget()  # Load the frame displayer
        self.stillimg = detection.stillgray
        self.blankimg = np.zeros_like(self.stillimg)
        # Start the timer for continuously processing the frame displayed in the GUI
        self.timer = QtCore.QTimer(parent=None)
        self.timer.timeout.connect(self.processFrame_slot)
        self.timer.start(400)
        self.count = 0  # Init second counter
        self.MouseAngle = 0
        self.covertic = time.perf_counter()
        self.cover_iter = 0
        self.ElisaGain = 1
        self.nploader = NpLoader()
        self.usbCam = UsbCamera()
        self.ueyeThread = UeyeThread()
        self.ueyeCam = self.ueyeThread.ueyeretriever
        self.detector = DetectionHelper()
        self.elisa3 = Elisa3Output()
        # self.odometryLoop = qt_classes.odometryLoop()
        self.GamePadThread = GamePadThread()
        # self.domain = detection.field_size_mm
        self.setupUiNew(MainWindow)
        self.state_dict = {}
        self.vizViewNumber = 0

    def PnPcallback(self, *args, **kwargs):
        """This callback tries to perform a PnP transformation on the image in the GUI"""
        np_data = self.imShow.np_data
        self.detector.tryPnP()
        self.detector.image_data.emit(np_data)

    def timerEvent(self, event, *args, **kwargs):
        if (event.timerId() != self.timer.timerId()):
            return
        self.processFrame_slot()

    def processFrame_slot(self, *args, **kwargs):
        """Processes np_data to a frame to show"""

        mtx = self.imShow.mtxrs
        dist = detection.dist
        mtxdetect, newmtxdetect, mtxwindow, newmtxwindow = self.imShow.getNewMtx()

        if self.rbDestNone.isChecked():
            processedFrame = self.blankimg
            self.imShow.showProcessedData_slot(processedFrame)
            return
        elif self.rbOverlayNone.isChecked():
            processedFrame = self.imShow.np_data.copy()
            self.imShow.showProcessedData_slot(processedFrame)
            self.timer.stop()
            # reset timer to updated slider position
            self.timer.start(int(self.frameRate ** -1 * 1000))
            return
        elif self.rbDestRaw.isChecked():
            processedFrame = self.imShow.np_data.copy()
            processedFrame = cv2.resize(self.imShow.np_data, self.imShow.showFrame.shape[1::-1])
            self.showRvec, self.showTvec, self.showMtx, self.showDist = self.detector.rvec, self.detector.tvec, mtxwindow, detection.dist
        elif self.rbDestUndistorted.isChecked():
            udFrame = np.zeros_like(self.imShow.np_data)
            # Do
            mtxdetect, newmtxdetect = detection.getCameraMatrices(self.imShow.np_data)
            mtxwindow, newmtxwindow = detection.getCameraMatrices(self.imShow.showFrame)
            # @param dst Output (corrected) image that has the same size and type as src .
            cv2.undistort(self.imShow.np_data,
                          mtxdetect,
                          detection.dist,
                          udFrame, newmtxdetect)
            processedFrame = cv2.resize(udFrame, self.imShow.showFrame.shape[1::-1])
            # cv2.aruco.getBoardObjectAndImagePoints()
            logger.debug("lalala")
            mtx = self.imShow.newmtx
            dist = None
            self.showRvec, self.showTvec, self.showMtx, self.showDist = self.detector.rvec, self.detector.tvec, newmtxwindow, None
        elif self.rbDestRectified.isChecked():
            # rectify the image:
            imShowmtx, imShowNewMtx = detection.getCameraMatrices(self.imShow.np_data)
            udFrame = np.zeros_like(self.imShow.np_data)

            cv2.undistort(self.imShow.np_data,
                          imShowmtx,
                          detection.dist,
                          udFrame,
                          imShowNewMtx)
            frame_in = cv2.resize(udFrame, self.imShow.showFrame.shape[1::-1])
            extra_mm_xy = np.array([90, 90])  # the width of the marker stick
            processedFrame = frame_in.copy()
            dom = np.array(self.detector.domain, np.float32)
            borderpix = bp = np.array([100, 100])

            overlays.rectifyCameraPose(frame_in,
                                       processedFrame,
                                       self.detector.rvec,
                                       self.detector.tvec,
                                       newmtxdetect,
                                       dom,
                                       border_pix=borderpix)
            # Compose the right matrices for showing a rectified frame with a border of bd [mm]:
            gamma = -500  # Gamma to adjust the depth of field of the projection.
            rv = np.array([0, 0, 0], np.float32)
            pp = np.array([[*bp * -1, 0], [*(dom + bp), 0]], np.float32)
            w, h = self.imShow.width, self.imShow.height
            w, h = dim = processedFrame.shape[1::-1]
            mtx = np.array(np.diag([*(dom + 2 * bp) ** -1, 1]) * np.diag([w, h, 1]), np.float32)
            ax, ay = dom
            bx, by = dim - borderpix * 2
            mtx[0, 0] = bx / ax * gamma
            mtx[1, 1] = by / ay * gamma
            mtx[2, 2] = 1
            mtx[0, 2] = cx = w / 2
            mtx[1, 2] = cy = h / 2
            tv = np.array([ax / bx * (bp[0] - cx), ay / by * (bp[1] - cy), gamma], np.float32)
            tv2 = np.array([ax / bx * (- cx), ay / by * (- cy), gamma], np.float32)
            self.showRvec, self.showTvec, self.showMtx, self.showDist = rv, tv, mtx, None
            p, _ = cv2.projectPoints(np.array([0, 0, 0], np.float32).reshape((1, 1, 3)), self.showRvec, self.showTvec,
                                     self.showMtx, self.showDist)
            logger.debug("lala")
        else:
            processedFrame = self.imShow.np_data.copy()
        # Convert to color:
        if len(processedFrame.shape) == 2:
            proc_image = cv2.cvtColor(processedFrame, cv2.COLOR_GRAY2BGR)

        elif len(processedFrame.shape) == 3:
            proc_image = processedFrame

        if self.chkCalMarkers.isChecked() or self.chkXvision.isChecked():
            # Check if there has been a detection step:
            if not self.detector.count:
                detection.addTextlines(self.imShow.np_data, [f"No detection step done yet"])
            else:
                corners, ids, rej = self.detector.rawCorners, self.detector.rawIds, self.detector.rawRej
                if self.chkCalMarkers.isChecked():
                    calCorners, calIds, _ = detection.filterMarkers(self.detector.rawCorners, self.detector.rawIds,
                                                                    detection.calibration_marker_ids)
                    pCorners = list(map(lambda x: cv2.projectPoints(x,
                                                                    self.showRvec,
                                                                    self.showTvec,
                                                                    self.showMtx,
                                                                    self.showDist)[0].reshape((1, 4, 2)).astype(
                        np.float32), self.detector.corners3D))

                    calCorners, calIds, _ = detection.filterMarkers(pCorners, self.detector.rawIds,
                                                                    detection.calibration_marker_ids)

                    # cv2.aruco.drawDetectedMarkers(proc_image,calCorners,calIds)
                    cv2.aruco.drawDetectedMarkers(proc_image, calCorners, calIds)

        if self.chkGrid.isChecked():
            overlays.drawDomain(proc_image, self.showRvec,
                                self.showTvec,
                                self.showMtx,
                                self.showDist, domain=self.detector.domain)
            logger.debug("Drawn calDomain")

        if self.chkCalstick.isChecked():
            detection.drawCalstick(proc_image, self.showRvec,
                                   self.showTvec,
                                   self.showMtx,
                                   self.showDist)
            logger.debug("Drawn calstick")

        # if self.chkGamePad.isChecked():
        #     controllers.showGamePadStates(proc_image, self.GamePadThread.gamePad.states)
        if self.rbControlMouse.isChecked():
            PTS = np.array(mousepos3d, np.float64).reshape((1, 1, 3))
            impoints, jac = cv2.projectPoints(PTS, self.showRvec, self.showTvec, self.showMtx, self.showDist)
            impoints = impoints.ravel()
            ptuple = tuple((int(impoints[0]), int(impoints[1])))

            cv2.drawMarker(proc_image,
                           ptuple,
                           (255, 122, 122),
                           cv2.MARKER_TILTED_CROSS,
                           20,
                           2,
                           cv2.LINE_AA)

        # if self.rbOverlayRotational.isChecked():
        # pos = QtGui.QCursor.pos()
        # poswin = self.imShow.parent().parent().parent().pos()
        # posimg = self.imShow.parent().pos()
        # posrel = pos - poswin - posimg
        # posxy = (posrel.x(), posrel.y() - 30)
        # try:
        #     pix = np.array(posxy, np.float32).reshape((1, 1, 2))
        #     pos3d = detection.correspondence2d3d(pix, self.showMtx,
        #                                          self.showDist,
        #                                          self.showRvec,
        #                                          self.showTvec, detection.calibration_markers_mm)
        #     # cv2.drawMarker(proc_image,posxy,(240,240,0),cv2.MARKER_TILTED_CROSS,markerSize=10)
        #     angle = self.MouseAngle
        #     model.Agent(states=[*pos3d.ravel()[:2], angle, 0, 0]).show_position_pnp(proc_image, self.showRvec,
        #                                                                             self.showTvec, self.showMtx,
        #                                                                             self.showDist)
        #     pWithAngle = [*pos3d.ravel()[:2], angle]
        #     e_ti, vabs = control_mapping.controlfield(pWithAngle, overlays.rotationalfield)
        #
        #     # self.lbCursorPos.setText(
        #     # f"{posxy} | {pos3d} | omega_i {omega_i:6.3f} e_ti {e_ti:6.3f} | zx {zx:6.3f} |zy {zy:6.3f}")
        # #
        #
        # except Exception as e:
        #     logger.error(e)
        if self.rbOverlayRotational.isChecked():
            overlays.drawRotation(proc_image, self.showRvec, self.showTvec, self.showMtx, self.showDist,
                                  self.detector.domain, invert=self.chkVizInvert.isChecked())
        if self.rvOverlayCoverage.isChecked():
            overlays.drawCoverage(proc_image, self.showRvec, self.showTvec, self.showMtx, self.showDist,
                                  self.detector.domain, viewnr=self.vizViewNumber, invert=self.chkVizInvert.isChecked())
            pstrl = []

            for i in range(len(gctronic.elisa_ids)):
                max_idx = cc.goals_max_index[i]
                max_path = cc.goal_paths[i, max_idx, :]
                detection.drawPath(max_path, proc_image, self.showRvec, self.showTvec, self.showMtx, self.showDist,
                                   color=(255, 0, 0))
                max_path = cc.goal_paths[i, max_idx, cc.goals_index[i]:]
                detection.drawPath(max_path, proc_image, self.showRvec, self.showTvec, self.showMtx, self.showDist,
                                   color=(0, 255, 0))
                traj = dwa.getPath(cc.states_all[i, :], dwa.config.predict_time, 4)
                trajpath = np.array(traj).T
                detection.drawPath(trajpath, proc_image, self.showRvec, self.showTvec, self.showMtx, self.showDist,
                                   color=(0, 0, 255), thickness=2)
        if self.chkXvision.isChecked():
            for agent in self.agents:
                agent.show_position_pnp(proc_image, self.showRvec, self.showTvec, self.showMtx, self.showDist)
        if self.chkXhat.isChecked():
            for agent in self.agents:
                agent.show_position_pnp(proc_image, self.showRvec, self.showTvec, self.showMtx, self.showDist,
                                        showOdometry=True)
        if self.chkXvision.isChecked():
            objpts = np.array(self.detector.corners3D).reshape(len(self.detector.corners) * 4, 3)
            projectedcorners = list(map(lambda x: cv2.projectPoints(x, self.showRvec, self.showTvec, self.showMtx,
                                                                    self.showDist)[0].reshape((1, 4, 2)).astype(
                np.float32), self.detector.corners3D))
            cv2.aruco.drawDetectedMarkers(proc_image, projectedcorners, self.detector.ids)
            logger.debug("pass")
        if self.chkReprojectionTest.isChecked():
            try:
                logger.debug("Testing if the corners are correcty used for pnp")
                logger.debug("Sort the values from config")
                sortedTpl2 = sorted(list(zip(detection.calibration_marker_ids, detection.calibration_marker_corners)),
                                    key=lambda x: x[0])

                [logger.debug(_) for _ in sortedTpl2]
                pnpIds2, pnpMarkerCorners = zip(*sortedTpl2)
                objpts = np.array(pnpMarkerCorners[:]).reshape(16, 3)

                impoints, jac = cv2.projectPoints(objpts, self.showRvec, self.showTvec, self.showMtx, self.showDist)
                detection.drawGridPoints(proc_image, impoints.reshape(4, 4, 2), drawCount=True, color=(69, 111, 243),
                                         drawCoords=False)
                calCorners3d, calIds, _ = detection.filterMarkers(self.detector.corners3D, self.detector.rawIds,
                                                                  detection.calibration_marker_ids)
                calCorners, calIds = self.detector.corners3D, self.detector.rawIds
                if len(calCorners):
                    sortedTpl = sorted(list(zip(calIds, calCorners)), key=lambda x: x[0].ravel())
                    pnpIds, pnpCorners = zip(*sortedTpl)
                    objpts = np.array(pnpCorners[:]).reshape(len(calCorners) * 4, 3)
                    impoints, jac = cv2.projectPoints(objpts - 50, self.showRvec, self.showTvec, self.showMtx,
                                                      self.showDist)
                    detection.drawGridPoints(proc_image, impoints.reshape(4, len(calCorners), 2), drawCount=True,
                                             color=(69, 255, 243),
                                             drawCoords=False)

            except Exception as e:
                logger.error("")

        self.imShow.showProcessedData_slot(proc_image)
        self.timer.stop()
        # reset timer to updated slider position
        self.timer.start(int(self.frameRate ** -1 * 1000))

    def setupUiNew(self, MainWindow):
        Ui_MainWindow.setupUi(self, MainWindow)
        # MainWindow.setObjectName("Pose Estimation")
        # Connect the button to load the camera parameters to the appropriate actions:
        self.btnCamParLoad.clicked.connect(self.loadCampars)
        self.btnReloadCfg.clicked.connect(self.reloadCfg)

        # self.GamePadThread.start()
        self.VideoFileStream = planebots.gui.qt_classes.videoFileReader(os.path.join("..", "img", "sample_video.mp4"))

        ## Connections:
        self.rbSourceStill.toggled.connect(self.nploader.loadStill)
        self.rbSourceAgent.toggled.connect(self.nploader.loadStillAgent)
        self.rbSourceUeye.toggled.connect(self.ueyeCam.toggle)
        self.rbSourceUeye.toggled.connect(self.imShow.updateDistortionCoeffs)
        self.rbSourceUsb.toggled.connect(self.usbCam.toggle)
        self.rbSourceVideo.toggled.connect(self.VideoFileStream.toggle)

        self.rbObsFuse.toggled.connect(self.resetFilters)
        self.rbObsVisUKF.toggled.connect(self.resetFilters)
        self.rbObsVis.toggled.connect(self.resetFilters)
        self.rbObsOdomUKF.toggled.connect(self.resetFilters)
        self.rbObsNone.toggled.connect(self.resetFilters)

        self.rbSourceVideo.toggled.connect(self.VideoFileStream.toggle)

        self.rbOutputElisa3.toggled.connect(self.elisa3.toggle)

        self.rbControlNone.toggled.connect(self.stopOutput)
        self.rbSourceStill.setChecked(1)
        self.rbDestRaw.setChecked(1)

        self.chkDetectLoop.toggled.connect(self.detector.toggle)
        self.chkDebug.toggled.connect(self.setDebugLogger)
        self.rbControlGamePad.toggled.connect(self.GamePadThread.toggle)
        # self.rbCGamePad.toggled.connect(self.GamePadThread.toggle)
        self.btnFolder.clicked.connect(self.openScreenShotFolder)
        self.btnElisaCalibrate.clicked.connect(self.calibrateOdometry)
        # self.r
        # Show fetched frames on screen:

        self.usbCam.image_data.connect(self.imShow.image_data_slot)
        self.ueyeCam.image_data.connect(self.imShow.image_data_slot)
        self.VideoFileStream.image_data.connect(self.imShow.image_data_slot)
        # Update detector frame
        self.usbCam.image_data.connect(self.detector.setDetectFrame_slot)
        self.ueyeCam.image_data.connect(self.detector.setDetectFrame_slot)
        self.VideoFileStream.image_data.connect(self.detector.setDetectFrame_slot)
        self.detector.storeGuiFrames.connect(self.dumpGuiFrame)
        self.GamePadThread.signal.connect(self.updateStatesFromGamepad)
        #
        # self.chkGamePad.toggled(self.)

        # self.nploader.image_data.connect(self.processFrame_slot)
        self.nploader.image_data.connect(self.imShow.image_data_slot)
        self.nploader.image_data.connect(self.processFrame_slot)
        self.nploader.image_data.connect(self.detector.setDetectFrame_slot)
        self.slideScreen.valueChanged.connect(self.setFramerateText_slot)
        self.slideDomainX.valueChanged.connect(self.setDomainText_slot)
        self.slideDomainY.valueChanged.connect(self.setDomainText_slot)
        self.slideDetect.valueChanged.connect(self.setDetectRate_slot)
        self.slideElisa.valueChanged.connect(self.setElisaRate_slot)
        self.slideMouseAngle.valueChanged.connect(self.setMouseAngle_slot)
        self.slideElisaGain.valueChanged.connect(self.setElisaGain_slot)
        self.slidePixClock.valueChanged.connect(self.setPixClock)
        self.slideFPS.valueChanged.connect(self.setFPS)
        self.slideExposure.valueChanged.connect(self.setExposure)
        # Connect button to the frame updater since it resets the timer
        self.btnFrameRate.clicked.connect(self.processFrame_slot)
        self.btnDetect.clicked.connect(self.detector.step)
        self.btnVizView.clicked.connect(self.incrementVizView)
        self.btnScreenShot.clicked.connect(self.imShow.dump)
        self.detector.detectionPerformed.connect(self.detectionStep_slot)
        self.btnPnP.clicked.connect(self.PnPcallback)
        self.detector.image_data.connect(self.detector.pnp)
        self.detector.detectionPerformed.connect(self.updateDetectionInfo_slot)
        self.detector.pnpTried.connect(self.updateDetectionPnPInfo_slot)
        self.chkPnP.toggled.connect(self.detector.togglePnP)
        self.btnSelectVideo.clicked.connect(self.setVideoSource)
        self.elisa3.measurement_update.connect(self.updateElisaMeasurement)
        self.btnMeas.clicked.connect(self.toggleMeasurement)
        self.btnOflip.clicked.connect(self.flipOval)
        self.btnVflip.clicked.connect(self.flipVval)
        self.btnPathRecalc.clicked.connect(self.pathRecalc)
        # Set the initial framerate to 10Hz:
        self.slideScreen.setValue(20)
        # Select screenshot as initial value
        # Defaults:
        self.detectionStep_slot()
        self.GamePadThread.signal.connect(self.updateStatesFromGamepad)
        self.ueyeThread.image_data.connect(self.imShow.image_data_slot)
        self.slideBeta.valueChanged.connect(self.setPathBeta_slot)
        # self.ueyeThread.run()
        if self.chkDebug.isChecked():
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def resetFilters(self, value):
        logger.info("Resetting filters")
        if value:
            for ag in self.agents:
                ag.filterv = ukFilter()
                ag.filtere = ukFilter()
                ag.filterv.x[:3] = [ag.z_vis[0] / 1000, ag.z_vis[1] / 1000, ag.z_vis[2]]
                ag.filtere.x[:3] = [ag.z_vis[0] / 1000, ag.z_vis[1] / 1000, ag.z_vis[2]]
                ag.filterv.predict(u=None, dt=0.1)  # Must be called before updates
                ag.filtere.predict(u=None, dt=0.1)  # Must be called before updates

    def dumpGuiFrame(self, savename):
        # pass
        cv2.imwrite(savename, self.imShow.showFrame)

    def reloadCfg(self):
        dwa.reload_config()
        logger.info("Dwa config and covercontrol package reloaded")
        reload(cc)
        reload(gctronic)
        reload(control_mapping)

    def incrementVizView(self):
        self.vizViewNumber += 1

    def pathRecalc(self):
        if self.rbControlCoverage.isChecked():
            cc.goal_paths = cc.goal_paths * np.nan

    def flipOval(self):
        self.slideOmega.setValue(-self.slideOmega.value())

    def flipVval(self):
        self.slideV.setValue(-self.slideV.value())

    def toggleMeasurement(self):
        if self.detector.storeData or self.elisa3.storeData:
            self.detector.storeData = False
            self.elisa3.storeData = False
            self.detector.makefolder()
            self.detector.saveTimeSteps()
            self.elisa3.saveTimeSteps()
            self.detector.measurementRunning = False

            self.btnMeas.setText("Start measurement")
        else:
            self.detector.storeData = True
            self.elisa3.storeData = True
            suffix = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            self.detector.storeTimeStepInit(self.detector.nMaxMeasurement, self.chkStoreVids.isChecked(), suffix=suffix)
            self.detector.measurementRunning = True
            self.elisa3.initNpz(suffix=suffix)
            self.btnMeas.setText("Stop measurement")
            logger.info("Starting detection measurement")

    def stopOutput(self, bool):
        if bool:
            for agent in self.detector.elisaList:
                agent.vleft = agent.vright = agent.states[3] = agent.states[4] = 0

    def calibrateOdometry(self):
        if self.elisa3.state:
            for agent in self.detector.elisaList:
                logger.info("Resetting the dead reakoning")
                self.elisa3.comm.library.calibrateSensors(agent.number)
                agent.odometry0 = agent.states[:3]

    def updateElisaMeasurement(self):
        """Updates when new data from the Elisa3 robots come in."""
        try:
            if self.rbObsOdomUKF.isChecked() or self.rbObsFuse.isChecked():
                for agent in self.agents:
                    ts = np.min([agent.dt, 0.5])
                    agent.filtere.predict(dt=ts)
                    uw = agent.z_uw
                    agent.filtere.update([uw[0], uw[1]], hx=planebots.control.observers.h_uw,
                                         R=np.eye(2) * [1, 1] * 1E-2)
                    agent.x_hat = agent.filtere.x
            if self.rbObsNone.isChecked():
                for agent in self.agents:
                    if agent.mid in gctronic.elisa_ids:
                        uw = [agent.z_uw[0], agent.z_uw[1]]
                        agent.x_hat = dwa.motion([*agent.x_hat[:3], *uw], dt=np.min([agent.dt, 0.3]), u=None)
        except np.linalg.LinAlgError as e:
            logger.error(e)

    def setDomainText_slot(self):
        self.lbDomX.setText(f" {self.slideDomainX.value()}mm")
        self.lbDomY.setText(f" {self.slideDomainY.value()}mm")

    def updateStatesFromGamepad(self, state_dict):
        self.state_dict.update(state_dict)

    def setDebugLogger(self, bool):
        logger = logging.getLogger("planebots")
        if not bool:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)

    def setVideoSource(self):
        selected = QtWidgets.QFileDialog.getOpenFileName()
        self.txtVideoFile.setText(selected[0])
        self.VideoFileStream.source.release()
        self.VideoFileStream.filepath = selected[0]
        self.VideoFileStream.toggle(True)

    def setFramerateText_slot(self, perc, *args, **kwargs):
        exponent = perc / 20 - 2
        ts = 10 ** exponent
        hz = ts ** -1
        self.lbFrameRate.setText(f"{perc}% | ^{exponent:4.2f} | Ts:{ts:4.2f}s | f={hz:4.2f}")
        self.frameRate = hz

    def setPathBeta_slot(self, val):
        value = float(val) / 10
        self.lbBeta.setText(f"Beta:{value:4.2f}")
        cc.beta = value

    def setElisaRate_slot(self, perc):
        exponent = perc / 20 - 3
        ts = 10 ** exponent
        hz = ts ** -1
        self.lbElisa.setText(f"Ts:{ts*1000:4.0f}ms | f={hz:4.2f}Hz")
        self.elisa3.setRate(hz)

    def setElisaGain_slot(self, perc):
        # 0-100 - 0 - 2
        exponent = perc / 50 / 99 * 100
        gain = 10 ** exponent
        self.ElisaGain = gain
        logger.info(f"Setting gain between 1 and 100: {gain:6.3f}")
        self.lbElisaGain.setText(f"{gain:4.2f}/100")

    def setDetectRate_slot(self, perc, *args, **kwargs):
        exponent = perc / 20 - 2
        ts = 10 ** exponent
        hz = ts ** -1
        # self.lbDetectRate.setText(f"{perc}% | ^{exponent:4.2f} | Ts:{ts:4.2f}s | f={hz:4.2f}")
        self.lbDetectRate.setText(f"Ts:{ts*1000:4.0f}ms | f={hz:4.2f}Hz")
        self.detector.setRate(hz)

    def setPixClock(self, slideVal, *args, **kwargs):
        self.lbPixelClock.setText(f"{slideVal}/71Mhz")
        self.ueyeCam.setPixelClock(slideVal)

    def setExposure(self, slideVal, *args, **kwargs):
        self.lbExposure.setText(f"{slideVal}ms")
        self.ueyeCam.setExposure(slideVal)

    def setFPS(self, slideVal, *args, **kwargs):
        actualFPS = self.ueyeCam.setFPS(slideVal)
        self.lbFPS.setText(f"{float(actualFPS):4.2f}/{float(slideVal):4.2f} FPS")
        # self.slideFPS.setValue(int(actualFPS))

    def setMouseAngle_slot(self, perc, *args, **kwargs):
        exponent = perc / 20 - 2
        ts = 10 ** exponent
        hz = ts ** -1
        angle = perc / 100 * np.pi * 2
        self.lbMouseAngle.setText(f"{perc}% | {angle:4.2f}")
        self.MouseAngle = angle
        self.detector.setRate(hz)

    def updateDetectionInfo_slot(self):
        showstr = "Elisa agents:\n"
        showstr += "\n".join([str(ag) for ag in sorted(self.agents, key=lambda x: x.mid)])
        self.lbElisaAgents.setText(showstr)
        # self.lb
        showstr = "calList:\n"
        showstr += "\n".join([str(ag) for ag in sorted(self.detector.calList, key=lambda x: x.mid)])
        showstr += "\nElisa:\n"
        showstr += "\n".join([str(ag) for ag in sorted(self.detector.elisaList, key=lambda x: x.mid)])
        showstr += "\nUnknown:\n"
        showstr += "\n".join([str(ag) for ag in sorted(self.detector.unknownList, key=lambda x: x.mid)])
        self.lbAgents.setText(showstr)

        countStr = f"Steps done:{self.detector.count}"

        tocStr = f"Detection time:{self.detector.analyzeTime*1000:6.1f} ms "
        tsStr = f"{self.detector.timeStamp}"
        self.lbDetectCount.setText("\n".join([countStr, tocStr, tsStr]))

        str1 = []

        if self.rbControlCoverage.isChecked():
            for i in range(len(gctronic.elisa_ids)):
                str1.append(
                    f"Agent {gctronic.elisa_ids[i]}/{gctronic.elisa_numbers[i]}: u {cc.states_all[i,3]:4.2f} w {cc.states_all[i,4]:4.2f}")
            parlist = 'max_accel max_yawrate max_speed min_speed predict_time speed_cost_gain robot_radius t_sim obstacle_avoidance v_n yawrate_n'.rsplit()
            plist = ["{1:<10}{0} ".format(_, getattr(dwa.config, _)) for _ in parlist]
            str1.extend(plist)
            parlist = 'decay gsx gsy max_num_goals max_path_points n_agents rho_max show_optimal_paths beta'.rsplit()
            plist = ["{1:<10}{0} ".format(_, getattr(cc, _)) for _ in parlist]
            str1.extend(plist)
        str1.append("\n".join([f"{k}:{self.state_dict[k]}" for k in self.state_dict]))

        self.lbControlTextBox.setText("\n".join(str1))
        if self.chkXvision.isChecked():
            dts = time.perf_counter() - self.detector.ts
            for ag in self.agents:

                if not np.isnan(self.detector.z_ids[ag.mid]):
                    xyth = self.detector.z_xyth[ag.mid]
                    ag.z_vis = xyth
                    ag.z_tsv = self.detector.ts
                    if self.rbObsVis.isChecked():
                        zv = [xyth[0] / 1000, xyth[1] / 1000, remap_angle(xyth[2])]
                        ag.x_hat[:3] = zv
                    elif self.rbObsVisUKF.isChecked():

                        zv = [xyth[0] / 1000, xyth[1] / 1000, remap_angle(xyth[2])]
                        ag.filterv.predict(np.min([dts, 0.3]))
                        ag.filterv.update(zv, hx=h_vision, R=np.eye(3) * [0.053, 0.053, 0.005])
                        ag.x_hat = ag.filterv.x
                    elif self.rbObsFuse.isChecked():
                        t_el = time.perf_counter() - ag.z_tsv
                        if t_el < 0.3:
                            zv = [xyth[0] / 1000, xyth[1] / 1000, xyth[2]]
                            ag.filtere.predict(np.min([dts, 0.3]))
                            ag.filtere.update(zv, hx=h_vision, R=np.eye(3) * [0.053, 0.053, 0.005])
                            ag.x_hat = ag.filtere.x
                    # else:
                    #     zv = [xyth[0]/1000,xyth[1]/1000,xyth[2]]
                    #     ag.x_hat[:3] = zv
                    # ag.updateFilterplanebots(xyth)
                else:
                    if self.rbObsVisUKF.isChecked():
                        ag.filterv.predict(np.min([dts, 0.3]))
                        ag.x_hat = ag.filterv.x
                logger.debug("Dumping z in agent objects")

    def updateDetectionPnPInfo_slot(self):
        self.lbRvec.setText(str(self.detector.rvec))
        self.lbTvec.setText(str(self.detector.tvec))
        pos, axs = detection.retrieveCamPos(self.detector.rvec, self.detector.tvec)
        txt = "pos: x:{0:+6.2f} y:{1:+6.2f} z:{2:+6.2f}\n".format(*list(pos.ravel()))
        txt += "rot: x:{0:+4.2f} y:{1:+4.2f} z:{2:+4.2f}".format(*axs.ravel())
        txt += f"\n Latest estimation succesfull?{self.detector.pnpSuccess }"
        txt += f"\n Continuous?{self.detector.continuousPnP }"
        self.lbPosAxs.setText(txt)

    def detectionStep_slot(self):
        np_data = self.imShow.np_data
        # mtx,newmtx,_,_ = self.imShow.getNewMtx()
        self.detector.domain = [self.slideDomainX.value(), self.slideDomainY.value()]
        cc.recalculate_domain(self.detector.domain, mm=True)
        self.detector.continuousPnP = self.chkPnP.isChecked()
        # try:
        # self.detector.step()
        if self.rbControlRotation.isChecked():
            agents = self.agents
            for agent in agents:
                # domain = detection.field_size_mm
                x = agent.x_hat
                x[2] = agent.x_hat[2]
                # x[2] = 0
                domain = self.detector.domain
                norm1 = np.array((x[0], x[1])) / domain * 1000
                e_ti, vabs = control_mapping.controlfield(x,
                                                          overlays.rotationalfield,
                                                          np.array(domain) / 1000)
                kv = control_mapping.potential_kv
                kw = control_mapping.potential_kw
                ke = self.ElisaGain

                agent.v = kv * ke
                agent.w = e_ti * kw * ke
                # agent.w = e_ti*np.pi**-1

                # agent.x_hat[3]= agent.v
                # agent.x_hat[4]= agent.w
                u1, u2 = gctronic.vtou(agent.v, agent.w)
                agent.vleft, agent.vright = int(u1), int(u2)

                # # Control mapping
                # ke = self.ElisaGain
                # v = ke*kv*vabs
                # v = 5
                # w = ke*kw*e_ti

                # omega_i = 5 * e_ti
                # l = 10 * (-1 * omega_i + .1)
                # l = 5 / (1 + 0 * e_ti) + 10 * omega_i
                # r = 5 / (1 + 0 * e_ti) - 10 * omega_i
                # # agent.vleft, agent.vright = int(r * self.ElisaGain), int(l * self.ElisaGain)
                # agent.v,agent.w = v,w
                # u1,u2 = gctronic.vtou(v,w)
                # agent.vleft,agent.vright = int(u1),int(u2)
                # agent.v,agent.w = gctronic.utov([agent.vleft,agent.vright])
                # states[i] = agent.states
        if self.rbControlCenter.isChecked():
            ctr = np.array([self.slideDomainX.value() / 2, self.slideDomainY.value() / 2]) / 1000
            for agent in self.agents:
                x = agent.x_hat
                u, cost = dwa.dwa_control([x[0], x[1], x[2], 0, 0], dwa.config, ctr.reshape(2, 1))
                agent.v = u[0]
                agent.w = u[1]
                u1, u2 = gctronic.vtou(*u)
                agent.vleft, agent.vright = int(u1), int(u2)

        if self.rbControlCustom.isChecked():
            agents = self.agents
            for j, agent in enumerate(agents):
                u = custom.c(j, agent.x_hat, dt=None)
                agent.v = u[0]
                agent.w = u[1]
                u1, u2 = gctronic.vtou(*u)
                agent.vleft, agent.vright = int(u1), int(u2)

        if self.rbControlMouse.isChecked():
            pos = QtGui.QCursor.pos()
            poswin = self.imShow.parent().parent().parent().pos()
            posimg = self.imShow.parent().pos()
            posrel = pos - poswin - posimg
            posxy = (posrel.x(), posrel.y() - 30)
            inplane = np.array([0, 0])
            global mousepos3d

            try:
                pix = np.array(posxy, np.float32).reshape((1, 1, 2))
                pos3d = detection.correspondence2d3d(pix, self.showMtx,
                                                     self.showDist,
                                                     self.showRvec,
                                                     self.showTvec, detection.calibration_markers_mm)
                x = np.clip(pos3d.ravel()[0] / 1000, cc.glx[0], cc.glx[-1])
                y = np.clip(pos3d.ravel()[1] / 1000, cc.gly[0], cc.gly[-1])
                inplane = np.array([x, y])
                mousepos3d = np.array([*inplane * 1000, detection.calibration_markers_mm[0, 2]], np.float32)
                # mousepos3d[:] = pos3d.ravel()
            except Exception as e:
                logger.error(e)
            for agent in self.agents:
                x = agent.x_hat
                u, cost = dwa.dwa_control(x, dwa.config, inplane.reshape(2, 1))
                agent.v = u[0]
                agent.w = u[1]
                u1, u2 = gctronic.vtou(*u)
                agent.vleft, agent.vright = int(u1), int(u2)

        if self.rbControlCoverage.isChecked():
            i = self.cover_iter
            agents = self.agents
            dt = time.perf_counter() - self.covertic
            self.covertic = time.perf_counter()

            cc.Z[:] = cc.Z * (1 - cc.decay * dt)

            for agent in agents[:cc.n_agents]:
                # cc.states_all
                # pos = agent.z_vis[:2]/1000
                pos = agent.x_hat[:3]
                idx = gctronic.elisa_ids.index(agent.mid)
                cc.states_all[idx, :3] = pos  # Update position of the correct agent
                # cc.states_all[idx,2] = agent.z_vis[2] # Update angle
            update_list = cc.control_step(self.cover_iter, min(dt, dwa.config.dt))
            if len(update_list):
                rv, update_list = cc.update_partition_and_goal(update_list)
                cc.calc_paths(update_list)
            for j in range(len(gctronic.elisa_ids)):
                cc.states_all_traj[j, i] = cc.states_all[j]
                x = cc.states_all[j]
                idxe = gctronic.elisa_ids[j]
                u1, u2 = gctronic.vtou(x[3], x[4], s=self.ElisaGain * 10)
                agents[j].v, agents[j].w = x[3], x[4]
                agents[j].vleft, agents[j].vright = int(u1), int(u2)
                # xn = dwa.motion(x, [x[3],x[4]], dt)  # simulate robot
                cc.states_all[j] = x
                cc.Z[:] = cc.add_coverage(cc.Z, x[:2], dt)

                # cc.Z[:] = cc.add_coverage(cc.Z, pos, dt)

                logger.debug("...")

        if self.rbControlGamePad.isChecked():
            try:
                agent = self.agents[self.state_dict['current_id']]
                agent.vleft = int(self.state_dict['l'] * self.slideElisaGain.value())
                agent.vright = int(self.state_dict['r'] * self.slideElisaGain.value())
                agent.v, agent.w = gctronic.utov([agent.vleft, agent.vright])
                self.agents[self.state_dict['current_id']].vright = int(
                    self.state_dict['r'] * self.slideElisaGain.value())
                prev_agent = self.agents[self.state_dict['previous_id']]
                prev_agent.vright = 0
                prev_agent.vleft = 0
                prev_agent.v = prev_agent.w = 0

            except KeyError as e:
                logger.error(e)
        if self.rbControlNone.isChecked():
            if hasattr(self, 'agents'):
                for agent in self.agents:
                    agent.vleft = agent.vright = agent.v = agent.w = 0
        if self.rbControlConstant.isChecked():
            for agent in self.agents:
                agent.vleft = int(self.slideV.value() + self.slideOmega.value())
                agent.vright = int(self.slideV.value() - self.slideOmega.value())
                agent.v, agent.w = gctronic.utov([agent.vleft, agent.vright])

        if self.rbInterDWA.isChecked():
            """Adjust the output to let the agents stay in the domain using the dwa method"""
            for agent in self.agents:
                x = [*agent.x_hat[:3], agent.v, agent.w]
                dw = dwa.calc_dynamic_window(x, dwa.interconfig)
                u_f, cost = dwa.calc_filter_input(agent.x_hat, dw, [x[3], x[4]], dwa.interconfig)
                lout, rout = gctronic.vtou(*u_f)
                agent.lout, agent.rout = lout, rout
        elif self.rbInterPredict.isChecked():
            for agent in self.agents:
                x = [*agent.x_hat[:3], agent.v, agent.w]
                dw = dwa.calc_dynamic_window(x, dwa.interconfig)
                u_f, cost = dwa.calc_filter_input_cont(agent.x_hat, dw, [x[3], x[4]], dwa.interconfig)
                agent.lout, agent.rout = gctronic.vtou(*u_f)
        else:
            for agent in self.agents:
                agent.lout, agent.rout = gctronic.vtou(agent.v, agent.w)

        self.lbCalMarkersId.setText("\n".join([f"{_}" for _ in detection.calibration_marker_ids]))
        self.lbCalMarkersPos.setText("\n".join([f"{_}" for _ in detection.calibration_markers_mm]))
        showstr = "Elisa agents:\n"
        showstr += "\n".join([str(ag) for ag in sorted(self.detector.agents, key=lambda x: x.mid)])
        self.lbElisaAgents.setText(showstr)
        # self.lb
        showstr = "\n".join([str(ag) for ag in sorted(self.detector.agentlist, key=lambda x: x.mid)])
        self.lbAgents.setText(showstr)
        self.updateDetectionPnPInfo_slot()
        self.lbDetectCount.setText(f"{self.detector.count} | {datetime.datetime.now()}")

        # except Exception as e:
        #     logger.error(e)
        #     raise e

    def dispayCamPose(self):
        self.lbRvec.setText()

    def changeImShow(self, npImage, *args, **kwargs):
        self.cnt += 1
        self.pushButton_2.setText(f"{self.cnt}")

    def loadCampars(self, event, *args, **kwargs):
        self.cnt += 1
        # self.btnClear.setText(f"{self.cnt}")
        self.btnCamParLoad.setText("Loading...")
        self.ueyeCam.camParLoad()

        self.btnCamParLoad.setText("Load Camera Parameters")
        # self.horizontalSlider.value()

    def openScreenShotFolder(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(planebots.packdir))


if __name__ == "__main__":
    import planebots

    # global agents
    logger = logging.getLogger("planebots")
    logger.addHandler(planebots.log_long)
    logger.setLevel(logging.DEBUG)
    ueye_log = logging.getLogger("planebots.ueye_camera")
    ueye_log.addHandler(planebots.log_long)
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow_wc(MainWindow)
    lbLogger = LogHook(ui.lbLog)
    logger.addHandler(lbLogger)
    ui.imageLayout.addWidget(ui.imShow)
    # Start detection loop:
    ui.detector.toggle(True)


    def btn_callback(MouseEvent):
        global cnt
        cnt += 1
        ui.cbOutputElisa.setText(f"{cnt}")


    ui.btnCamParLoad.show()
    MainWindow.show()

    logger.info("Exiting..")
    sys.exit(app.exec_())
