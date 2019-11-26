import datetime
import logging
import os

import cv2
import numpy as np

import planebots
from planebots.vision import detection

logger = logging.getLogger(__name__)
# calibration_markers_orientation = json.loads(planebots.config.get("detection", "calibration_markers_orientation"))
# calibration_markers_size = 80


if __name__ == '__main__':
    logger = logging.getLogger("planebots")
    logger.addHandler(planebots.log_long)
    logger.setLevel(logging.DEBUG)
    logger.info("Starting script")
    mname = os.path.abspath(os.path.join(planebots.packdir, 'img', "marker00.png"))
    mframe = cv2.imread(mname)
    corners, ids, rej = cv2.aruco.detectMarkers(mframe, planebots.MarkerDictionary)

    videoname = os.path.abspath(os.path.join(planebots.packdir, 'stills', "line_far_highres.mp4"))
    # videoname = os.path.abspath(os.path.join(planebots.packdir, 'stills', "500-300-1000-Highres.mp4"))
    logger.info(f"Opening {videoname}")
    stream = cv2.VideoCapture(videoname)
    ret, frame = stream.read()
    corners, ids, rej = cv2.aruco.detectMarkers(frame,
                                                planebots.MarkerDictionary,
                                                None,
                                                None,
                                                planebots.DetectionParameters,
                                                None)
    mtxWindow, newMtxWindow = detection.getCameraMatrices(frame)

    cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(255, 122, 0))
    detection.addTextlines(frame, [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    cv2.imwrite("testPnP.png", frame)

    f2 = cv2.undistort(frame, mtxWindow, detection.dist, None, mtxWindow)

    detection.addTextlines(f2, [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    cv2.imwrite("testPnP2.png", f2)
    f2 = cv2.undistort(frame, mtxWindow, detection.dist, None, newMtxWindow)
    detection.addTextlines(f2, [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    cv2.imwrite("testPnP3.png", f2)
    pnpSucces, rvec, tvec = detection.findPnP(mtxWindow, detection.dist, corners, ids, detection.calibration_marker_ids,
                                              detection.calibration_markers_mm, calcError=True)

    ref3d = np.array(detection.calibration_markers_mm, np.float64)
    impoints, jac = cv2.projectPoints(ref3d, rvec, tvec, mtxWindow, detection.dist)
    detection.drawPoly(frame, impoints)
    for ref in detection.calibration_marker_corners:
        impoints, jac = cv2.projectPoints(ref, rvec, tvec, mtxWindow, detection.dist)
        detection.drawPoly(frame, impoints, closed=False, color=(255, 122, 60))
    # detection.drawFrameGrid()
    # detection.drawGridPoints(frame, impoints)

    cv2.imwrite("testPnP_reprojection.png", frame)

    calcorners, calids, missingids = detection.filterMarkers(corners, ids, detection.calibration_marker_ids)
    sortedTpl = sorted(list(zip(calids, calcorners)), key=lambda x: x[0].ravel())
    sortedTpl2 = sorted(list(zip(detection.calibration_marker_ids, detection.calibration_marker_corners)),
                        key=lambda x: x[0])
    pnpIds, pnpCorners = zip(*sortedTpl)
    pnpIds2, pnpMarkerCorners = zip(*sortedTpl2)
    objpts = np.array(pnpMarkerCorners[:]).reshape(16, 3)
    impoints, jac = cv2.projectPoints(objpts, rvec, tvec, mtxWindow, detection.dist)

    detection.drawGridPoints(frame, np.array(pnpCorners).reshape((4, 4, 2)), drawCount=True, color=(69, 243, 111),
                             drawCoords=False)
    detection.drawGridPoints(frame, impoints.reshape(4, 4, 2) - 120, drawCount=True, color=(69, 111, 243),
                             drawCoords=False)
    # cv2.imwrite("kdksdkks.png",frame)

    rv, rvec2, tvec2 = cv2.solvePnP(objpts, np.array(pnpCorners).reshape(16, 2), mtxWindow, detection.dist,
                                    cv2.SOLVEPNP_ITERATIVE)
    repr, jac = cv2.projectPoints(objpts, rvec2, tvec2, mtxWindow, detection.dist)
    reperr = repr - np.array(pnpCorners).reshape((16, 1, 2))
    total_error = np.linalg.norm(reperr) / 4

    cv2.imwrite("kdksdkks.png", frame)
    impointsIterative, jac = cv2.projectPoints(objpts, rvec2, tvec2, mtxWindow, detection.dist)
    detection.drawGridPoints(frame, impointsIterative.reshape(4, 4, 2), drawCount=True, color=(69, 0, 243),
                             drawCoords=False)

    cv2.imwrite("kdksdkks2.png", frame)

    logger.debug(pnpSucces)
