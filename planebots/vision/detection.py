import datetime
import json
import logging
import os
import time

import cv2
import numpy as np

import planebots
from planebots.vision import ueye_camera
from planebots.vision.ueye_camera import cameraNewFrame

logger = logging.getLogger(__name__)
calibration_markers_mm = np.array(json.loads(planebots.config.get("detection", "calibration_markers_mm")), np.float64)
calibration_outline_mm = np.array(json.loads(planebots.config.get("detection", "calibration_outline_mm")), np.float64)
calibration_marker_ids = json.loads(planebots.config.get("detection", "calibration_marker_ids"))
detection_limiter_hz = json.loads(planebots.config.get("detection", "limiter_hz"))
stillgray = cv2.imread(os.path.join("..", "img", "still.png"), cv2.IMREAD_GRAYSCALE)
stillagent = cv2.imread(os.path.join("..", "img", "stillagent.png"), cv2.IMREAD_GRAYSCALE)
elisa_ids = json.loads(planebots.config.get("gctronic", "elisa_ids"))
field_size_mm = json.loads(planebots.config.get("detection", "field_size_mm"))
mtx, dist, _, _ = planebots.calibration_gigueye
original_size = (1280, 1024)
calibration_markers_orientation = json.loads(planebots.config.get("detection", "calibration_markers_orientation"))
calibration_marker_size = json.loads(planebots.config.get("detection", "marker_size"))
calibration_clockwise = planebots.config["detection"].getboolean("calibration_clockwise")


def getMarkerCorners(coord3d, size, angle, clockwise=calibration_clockwise):
    a = angle / 180 * np.pi + np.pi
    if not clockwise:
        a = -a
    a = -a
    mg = np.meshgrid([1, -1], [-1, 1], 0)
    flat = np.array(mg, np.float64).reshape(3, 4).T
    flat[0, :], flat[1, :] = flat[1, :], flat[0, :].copy()
    if clockwise:
        flat[1, :], flat[3, :] = flat[3, :], flat[1, :].copy()

    rm = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    corners = flat.dot(rm * size / 2) + coord3d

    logger.debug("done")
    return corners


def getCalibrationVertices(calibration_markers_mm, calibration_markers_size, calibration_markers_orientation):
    input = list(zip(calibration_markers_mm, [calibration_markers_size] * 4, calibration_markers_orientation))
    rv = list(map(lambda x: getMarkerCorners(*x), input))
    return rv


calibration_marker_corners = getCalibrationVertices(calibration_markers_mm, calibration_marker_size,
                                                    calibration_markers_orientation)


def getCameraMatrices(frame):
    """Calculates camera matrices dependent on size of the frame retrieved."""
    framesize = frame.shape[1::-1]
    mtxwindow, roi = cv2.getOptimalNewCameraMatrix(mtx, 0, original_size, 1, framesize)
    newmtxwindow, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, original_size, 1, framesize)
    return mtxwindow, newmtxwindow


def table(ids, pixelcoord, pos3d, angle):
    """Convenience fcn"""
    "{:<4} | {:>10} | {:<20}| {:<50}".format()


def addTextlines(frame, lines=[], alpha=0.7, copyright=True, pos='ul', offset=(0, 0), boxcolor=(0, 0, 0), **kwargs):
    """Add textline in the upperleft corner of a frame with black background
    alpha 0: no transparancy 1: fully opaque"""
    # Generate a blank, colored image in the same size as the parent image:
    blank_image = np.ones((300, 300, 3), np.uint8)
    dims = frame.shape
    logger.debug(dims)
    if len(dims) == 2:
        black = cv2.resize(blank_image.copy(), dims)
    else:
        dims = frame.shape[::-1][1:]
        black = cv2.resize(blank_image.copy(), frame.shape[::-1][1:])

    # DimensionsL
    lineheight = 15
    charwidth = 10
    indent = 10
    frameheight = dims[0]
    framewidth = dims[1]
    typesets = {'fontFace': 1, 'fontScale': 1, 'color': (255, 255, 255), 'thickness': 1}
    typesets.update(**kwargs)
    # black = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)
    framerects = frame.copy()
    if type(lines) != type([]):
        lines = [lines]
    n_lines = len(lines)

    if pos == 'll' or pos == 'lr':
        lines = list(reversed(lines))

    for n, line in enumerate(lines):
        line = str(line)
        linewidth = len(line) * charwidth + indent
        # set the bounds for textarea:
        if pos == 'ul':
            lb, ro = (0, n * lineheight), (linewidth, (n + 1) * lineheight)
        elif pos == 'll':
            lb, ro = (0, frameheight - (n + 2) * lineheight), (linewidth, frameheight - (n + 1) * lineheight)
        elif pos == 'lr':
            lb, ro = (framewidth - linewidth, frameheight - (n + 2) * lineheight), (
                framewidth, frameheight - (n + 1) * lineheight)
        lb = (lb[0] + offset[0], lb[1] + offset[1])
        ro = (ro[0] + offset[0], ro[1] + offset[1])

        # white = blank_image = np.zeros(frame.shape[::-1], np.uint8)
        logger.debug(frame.shape[::-1][1:])
        # blank_image_gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY) #convert color to gray

        cv2.rectangle(framerects, pt1=lb, pt2=ro, color=0, thickness=-1)
        # cv2.rectangle(frame,pt1=lb,pt2=ro,color=0,thickness=-1)
        textargs = {"text": line, 'org': (lb[0] + indent, ro[1] - 2)}
        typesets.update(textargs)
        cv2.putText(framerects, **typesets)
        cv2.putText(frame, **typesets)
    if copyright:
        message = b"0xC2 A.R. Vermeulen".decode('utf-8')
        lineheight = 8
        charwidth = 5
        indent = 5
        linewidth = (len(message) * charwidth + indent)
        lb, ro = (0, frameheight - lineheight), (linewidth, frameheight)
        cv2.rectangle(framerects, pt1=lb, pt2=ro, color=boxcolor, thickness=-1)

        textargs = {'fontScale': 0.5, "text": message, 'org': (indent, ro[1] - 2)}
        typesets.update(textargs)
        cv2.putText(framerects, **typesets)
        cv2.putText(frame, **typesets)
    white = cv2.bitwise_not(black)
    cv2.addWeighted(frame, alpha, framerects, 1 - alpha, 0, frame)
    return frame


def drawCircles(frame, points, **kwargs):
    LineArgs = {'color': (122, 122, 122),
                'thickness': 2,
                'radius': 10,
                'lineType': cv2.LINE_AA}
    LineArgs.update(**kwargs)

    LinePoints = np.squeeze(points)
    for i in range(len(LinePoints)):
        p1 = tuple(map(lambda x: int(x), [*LinePoints[i]]))
        cv2.circle(frame, p1, **LineArgs)


class NoMarkersException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return "No markers detected!"


def getPositionsFast(image, rvec, tvec, mtxframe, mtxshow, ref3d, dist, detectionParameters):
    # get the raw corners of the image:
    corners, ids, rej = cv2.aruco.detectMarkers(image,
                                                planebots.MarkerDictionary,
                                                None,
                                                None,
                                                detectionParameters,
                                                None)
    # Undistort with the correct mtx for the detection size:
    cornerspts = list(map(lambda x: cv2.undistortPoints(x, mtxframe, dist, None, None, mtxshow), corners))
    rejpts = list(map(lambda x: cv2.undistortPoints(x, mtxframe, dist, None, None, mtxshow), rej))
    # Calculate the position of the corners in the image plane:

    cornersInPlane = list(
        map(lambda x: correspondence2d3d(x, mtxframe, dist, rvec, tvec, calibration_markers_mm), corners))

    centersInPlane, anglesInPlane = centerMarkers(cornersInPlane)
    detection_states = {}
    detection_states["corners"] = corners
    detection_states["ids"] = ids
    detection_states["rej"] = rej
    detection_states["rejpts"] = rejpts
    detection_states["cornerspts"] = cornerspts
    detection_states["centersInPlane"] = centersInPlane
    detection_states["anglesInPlane"] = anglesInPlane
    detection_states["success"] = True
    return centersInPlane, anglesInPlane, ids, detection_states


def toTuples(centersInPlane, anglesInplane, ids):
    tuples = list(zip(centersInPlane, anglesInplane, ids))
    return tuples


def toDict(centersInPlane, anglesInplane, ids):
    tuples = list(zip(centersInPlane, anglesInplane, ids))
    dictList = []
    for entry in tuples:
        d = {}
        d.update({"Position"})
        dictList.append()
    return tuples


def findPnP(mtx, dist, corners, ids, nodelist, ref3d, calcError=True):
    """Find the Perspective and pose with use of the corners of the detected markers"""
    try:

        if not len(corners):
            raise NoMarkersException

        calcorners, calids, missingids = filterMarkers(corners, ids, nodelist)
        logger.debug("Sorting Markers...")
        sortedRefs = sorted(list(zip(nodelist, ref3d)), key=lambda x: np.array(x[0]).ravel())
        sortedCalCorners = sorted(list(zip(calids, calcorners)), key=lambda x: x[0].ravel())
        logger.debug("Sorted calibration markers and 3d positions:")
        [logger.debug(_) for _ in sortedRefs]

        if len(sortedRefs) == len(sortedCalCorners):
            # Whe have found all calibration markers!
            pnpIds, pnpCorners = zip(*sortedCalCorners)
            pnpIds2, pnpRef3d = zip(*sortedRefs)
            centers, angles = centerMarkers(pnpCorners)
            tic = time.perf_counter()
            rv, rvec, tvec = cv2.solvePnP(np.array(pnpRef3d, np.float64), centers, mtx, dist,
                                          flags=cv2.SOLVEPNP_P3P)  # The magic happens here!
            toc = time.perf_counter()
            if calcError:
                # Reprojection error:
                repr, jac = cv2.projectPoints(np.array(pnpRef3d, np.float64), rvec, tvec, mtx, dist)
                reperr = repr - centers
                total_error = np.linalg.norm(reperr, axis=2)
                logger.debug(f"Norm of total error/datapoints using the centers: {total_error/4} in {toc-tic:06.5f}s")

                rv, rvecs, tvecs = cv2.solveP3P(np.array(pnpRef3d[1:4], np.float64), centers[1:4], mtx, dist,
                                                cv2.SOLVEPNP_P3P)
                rv, rvecs, tvecs = cv2.solveP3P(np.array(pnpRef3d[:3], np.float64), centers[:3], mtx, dist,
                                                cv2.SOLVEPNP_P3P)
                for i in range(len(rvecs)):
                    # Reprojection error:
                    repr, jac = cv2.projectPoints(np.array(pnpRef3d, np.float64), rvecs[i], tvecs[i], mtx, dist)
                    reperr = repr - centers
                    total_error = np.linalg.norm(reperr)
                    logger.debug(f"Norm of total error: {total_error/4}")
                    if total_error >= 100:
                        logger.warning(f"Norm of total error: {total_error}")

            # Third method:
            sortedTpl = sorted(list(zip(calids, calcorners)), key=lambda x: x[0].ravel())
            sortedTpl2 = sorted(list(zip(calibration_marker_ids, calibration_marker_corners)), key=lambda x: x[0])
            pnpIds, pnpCorners = zip(*sortedTpl)
            pnpIds2, pnpMarkerCorners = zip(*sortedTpl2)
            objpts = np.array(pnpMarkerCorners[:]).reshape(16, 3)

            rv, rvec2, tvec2 = cv2.solvePnP(objpts, np.array(pnpCorners).reshape(16, 2), mtx, dist,
                                            cv2.SOLVEPNP_ITERATIVE)

            repr, jac = cv2.projectPoints(objpts, rvec2, tvec2, mtx, dist)
            reperr = repr - np.array(pnpCorners).reshape(16, 1, 2)
            total_error = np.linalg.norm(reperr) / 4

            if rv:
                if total_error >= 1000:
                    # logger.debug(f"Total error of pnp too great {total_error}")
                    logger.warning(f"Total error of pnp too great {total_error}")
                    return False, None, None
                pnpSucces = True
                return pnpSucces, rvec2, tvec2
        else:
            return False, None, None

    except (ValueError, AttributeError) as e:
        # raise planebots.detection.CalibrationDetectionError(calids)
        logger.error(e)
        pnpSucces = False
        rvec = tvec = None
    return pnpSucces, rvec, tvec


def drawPoly(frame, points, closed=True, **kwargs):
    """
    Draws polygon by connecting the Nx2 points with lines
    :param frame:
    :param points:
    :param color:
    :param closed:
    :return:
    """
    LineArgs = {'color': (122, 122, 122),
                'thickness': 2,
                'lineType': cv2.LINE_AA}
    LineArgs.update(**kwargs)
    LinePoints = np.squeeze(points)
    n, m = LinePoints.shape
    for i in range(len(LinePoints) - 1):
        lpi = LinePoints[i]
        p1 = tuple(map(lambda x: int(x), [*LinePoints[i]]))
        p2 = tuple(map(lambda x: int(x), [*LinePoints[i + 1]]))
        cv2.line(frame, p1, p2, **LineArgs)
    if closed:
        p1 = tuple(map(lambda x: int(x), [*LinePoints[0]]))
        p2 = tuple(map(lambda x: int(x), [*LinePoints[-1]]))
        cv2.line(frame, p1, p2, **LineArgs)


def interpolatePoints(points, n=5):
    tPoints = np.array(points, np.float32)
    interpolation_factor = n
    tNew = np.zeros(np.shape(tPoints) * np.array([interpolation_factor, 1]))
    for j in range(len(tPoints)):
        for i in range(interpolation_factor):
            p1 = tPoints[(j + 1) % len(tPoints)]
            p2 = tPoints[j]
            ip = p1 * i / interpolation_factor + p2 * (-i + interpolation_factor) / interpolation_factor
            idx = interpolation_factor * j + i
            tNew[idx] = ip

    return tNew


def drawCalstick(frame, rvec, tvec, mtx, dist):
    tPoints = interpolatePoints(np.array(calibration_outline_mm, np.float32), 10)
    # Bottom points:
    if calibration_clockwise:
        bPoints = tPoints + [0, 0, -15]
    else:
        bPoints = tPoints + [0, 0, 15]

    # Draw upper points:
    pts, jac = cv2.projectPoints(tPoints, rvec, tvec, mtx, dist)
    drawPoly(frame, pts, True, color=(125, 75, 88))
    # Draw lower points:
    pts, jac = cv2.projectPoints(bPoints, rvec, tvec, mtx, dist)

    drawPoly(frame, pts, True, color=(125, 75, 88))

    # Draw connections:
    for i in range(len(tPoints)):
        pts, jac = cv2.projectPoints(np.array([tPoints[i], bPoints[i]], np.float32), rvec, tvec, mtx, dist)
        drawPoly(frame, pts, True, color=(125, 75, 88))


def drawPath(path, frame, rvec, tvec, mtx, dist, mm=False, height=0, **kwargs):
    nn = np.where(np.isnan(path[:, 0]) == False)
    ppoints = path[nn, :]
    if ppoints.size > 2:
        ppoints3d = np.zeros((ppoints.size // 2, 3))
        ppoints3d[:, :2] = ppoints
        ippoints = ppoints3d
        if not mm:
            ippoints = ippoints * 1000

        if calibration_clockwise:
            ippoints = ippoints + [0, 0, -height]
        else:
            ippoints = ippoints + [0, 0, height]
        # Draw upper points:
        pts, jac = cv2.projectPoints(ippoints, rvec, tvec, mtx, dist)
        opts = {"color": (255, 0, 0)}
        opts.update(kwargs)
        drawPoly(frame, pts, closed=False, **opts)


def drawGridPoints(frame, points, color=(122, 122, 122), drawCrosses=True, drawCoords=True, drawCount=False,
                   drawLines=True, **kwargs):
    """Size of points is nxmx2"""
    n, m, o = points.shape
    cnt = 0
    for i in range(n):
        linepoints = points[i, :, :]
        for j in range(m):
            ptuple = tuple((int(points[i, j, 0]), int(points[i, j, 1])))
            if drawCrosses:
                cv2.drawMarker(frame,
                               ptuple,
                               color,
                               cv2.MARKER_TILTED_CROSS,
                               20,
                               2,
                               cv2.LINE_AA)
            txt = ""
            if drawCount:
                txt += f"{cnt}"
                cnt += 1
            if drawCoords:
                txt += f" {ptuple}"
            cv2.putText(frame,
                        txt,
                        ptuple,
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        color,
                        2,
                        cv2.LINE_AA)
    if drawLines:
        for i in range(n):
            linepoints = points[i, :, :]
            linetuples = list(map(lambda pt: tuple((int(pt[0]), int(pt[1]))), linepoints))
            for j in range(len(linepoints) - 1):
                p1 = linetuples[j]
                p2 = linetuples[j + 1]
                # color= (255//n*i,255-255//n*i,255)
                try:
                    cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
                except OverflowError as e:
                    logger.error(e)
        for j in range(m):
            vertLinePoints = points[:, j, :]
            vertLineTuples = list(map(lambda pt: tuple((int(pt[0]), int(pt[1]))), vertLinePoints))
            for i in range(len(vertLinePoints) - 1):
                try:
                    p1 = vertLineTuples[i]
                    p2 = vertLineTuples[i + 1]

                    cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
                except OverflowError as e:
                    logger.error(e)


def filterMarkers(corners, ids, idselect):
    """
    :param corners: corners from cv2.aruco.DetectMarkers
    :param ids: ids from cv2.aruco.DetectMarkers
    :param idselect: ids to be selected
    :return:
    """
    detCorners = []
    detIds = []
    missingIds = []

    if len(corners) == 0:
        return [], np.array([]), idselect

    idselectflat = np.ravel(idselect)

    for k, v in enumerate(ids.ravel()):
        if v in idselectflat:
            detCorners.append(corners[k])
            detIds.append(v)

    missingIds = list(set.difference(set(detIds), set(idselectflat)))
    return detCorners, np.array([detIds]).T, np.array([missingIds]).T


def getAngle(corners):
    x, y = corners[0].T
    x0 = x[0]
    y0 = y[0]
    angle1 = np.mod(-np.arctan2(x0 - x[2], y0 - y[2]) + np.pi / 4 + np.pi, np.pi * 2)
    deg = angle1 / np.pi * 180
    return angle1


def centerMarkers(corners):
    angles = np.zeros((len(corners)), np.float32)
    for i, cr in enumerate(corners):
        c = cr[0, :, :2]
        n1 = c[1] - c[0]
        n2 = np.array([1, 1])
        angle = getAngle(cr[:, :, :2])
        angles[i] = angle
    return np.array([np.average(corner, 1) for corner in corners]), angles


def retrieveCamPos(rvec, tvec):
    logger.debug("Retrieving camera positions")
    rm, jcb = cv2.Rodrigues(rvec)
    rmupper = np.concatenate((rm.T, np.matmul(rm.T, tvec)), 1)
    rmlow = np.matrix('[0 0 0 1]')
    rminv = np.concatenate((rmupper, rmlow), 0)
    campos = rminv * np.matrix('[0 0 0 1]').T
    zaxis = rminv * np.matrix('[0 0 1 0]').T
    # https://math.stackexchange.com/questions/82602/how-to-find-camera-position-and-rotation-from-a-4x4-matrix
    campos2 = np.dot(-rm.T, tvec)
    z2 = np.dot(rm.T, np.matrix('[0 0 1]').T)
    pos, axs = campos2, np.array(z2)
    txt = "pos: x:{0:04.2f} y:{1:04.2f} z:{2:04.2f}".format(*list(pos.ravel()))
    txt += "rot: x:{0:04.2f} y:{1:04.2f} z:{2:04.2f}".format(*axs.ravel())
    logger.debug(txt)
    return pos, axs


def get3dref(detectedIds, calIds, calPositions):
    detectedIdsFlat = detectedIds.ravel()

    def sortfun(x):
        try:
            idx = list(detectedIdsFlat).index(calIds[x[0]])
        except ValueError as e:
            idx = 9E9
        return idx

    calPositionsShuffled = sorted(enumerate(calPositions), key=sortfun)
    order, calPositionsSorted = zip(*calPositionsShuffled)
    ref3d = np.array([calPositionsSorted[:len(detectedIds)]], np.float64)
    return ref3d


def getRefFromConfig(calids):
    calidsflat = calids.ravel()
    p3dlist = json.loads(planebots.config.get("video_processing", "calibration_markers_mm"))
    refIds = json.loads(planebots.config.get("video_processing", "calibration_marker_ids"))

    # refIds = calIds

    def sortfun(x):
        idx = list(calidsflat).index(refIds[x[0]])
        return idx

    try:
        p3dlist_shuffled = sorted(enumerate(p3dlist), key=sortfun)
    except ValueError as e:
        raise CalibrationDetectionError(calids)
    order, p3d_sorted = zip(*p3dlist_shuffled)
    ref3d = np.array([p3d_sorted], np.float64)
    return ref3d


def drawFrameGrid(frame, n=5, m=6, **kwargs):
    x, y = np.meshgrid(np.linspace(0, frame.shape[1], n), np.linspace(0, frame.shape[0], m))
    gridpoints = np.array([x, y]).swapaxes(0, 2)
    drawGridPoints(frame, gridpoints, color=(122, 122, 122), drawCrosses=False, drawCoords=False, **kwargs)


def correspondence2d3d(xypix, cameraMatrix, distCoeffs, rvec, tvec, planepoints):
    """Solving the 2d3d correspondence by projecting the possible solutions on the domain plane"""
    logger.debug("Solving the correspondence2d3d")
    shape = xypix.shape
    undistortSettings = {'cameraMatrix': cameraMatrix, 'distCoeffs': distCoeffs, 'dst': None, 'R': None,
                         'P': cameraMatrix}
    xypix_ud = cv2.undistortPoints(xypix, **undistortSettings)

    # xypix = np.mat("[0;0;1]")
    xypix = xypix_ud

    pvec = np.mat(planepoints[:3]).T
    # pvec=np.mat("[0 1 0; 0 0 1; 0 0 0]")
    R, jac = cv2.Rodrigues(rvec)  # Retrieve the 3x3 rotation matrix
    t = tvec.reshape((3, 1))  # Cast the translation vector in the correct dimensions
    rta = np.matmul(-R.T, np.linalg.inv(cameraMatrix))
    rtt = np.matmul(-R.T, t)
    cl2 = pvec[:, 1] - pvec[:, 0]
    cl3 = pvec[:, 2] - pvec[:, 0]
    b = -pvec[:, 0] - rtt

    xyzarr = []
    for i, xy in enumerate(cv2.convertPointsToHomogeneous(xypix)):
        xyhom = np.mat(xy).T
        cl1 = np.matmul(rta, xyhom)
        a = np.concatenate((cl1, cl2, cl3), 1)
        X = np.linalg.solve(a, b)
        xyz = pvec[:, 0] + cl2 * X[1] + cl3 * X[2]

        xyzarr.append(xyz)
    xyznparr = np.array(xyzarr)
    rv = xyznparr.reshape((*shape[:-1], 3))

    # logger.debug("correspondence2d3d")
    return -rv


class CalibrationDetectionError(Exception):
    def __init__(self, ids, *args):
        self.ids = ids
        self.message = f'Less than 4 calibration ids detected: {len(self.ids)}: {np.array(self.ids).ravel()}'  # without this you may get DeprecationWarning
        # Special attribute you desire with your Error,


class PoseAmbiguityError(Exception):
    def __init__(self, rvecs, tvecs, *args):
        self.rvecs = rvecs
        self.tvecs = tvecs

        self.message = f'3 calibration ids detected without reference pose!'  # without this you may get DeprecationWarning
        # Special attribute you desire with your Error,


def saveFrame(originalFrame, prefix="", suffix=None):
    if suffix == None:
        suffix = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S.%f")
    return cv2.imwrite(os.path.join(planebots.packdir, "stills", f"{prefix}frame_{suffix}.png"), originalFrame)


def getFrame(frame, source):
    if source == "still":
        frame = stillgray.copy()
        ret = 1
    elif source == "ueye":
        ret, frame = cameraNewFrame(frame, ueye_camera.camera_online)
    return ret, frame


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.addHandler(planebots.log_long)
    logger.setLevel(logging.DEBUG)
    videoname = os.path.abspath(os.path.join('stills', "lower_left_lowres.mp4"))
    logger.info(f"Opening {videoname}")
    stream = cv2.VideoCapture(videoname)
    ret, frame = stream.read()

    calPositions = json.loads(planebots.config.get("video_processing", "calibration_markers_mm"))
    calIds = json.loads(planebots.config.get("video_processing", "calibration_marker_ids"))
    # ret, frame = stream.read()
    # rvec = None
    # tvec = None
    mtx, dist, _, _ = planebots.calibration_gigueye
    # positions, _, _, rvec, tvec = getPositions(frame, mtx, dist, planebots.MarkerDictionary, planebots.DetectionParameters,
    #                                            calIds, calPositions, rvec, tvec)
    # pts, jac = cv2.projectPoints(positions, rvec, tvec, mtx, dist)
    # drawGridPoints(frame, pts)
    #
    # # defining the corner points
    # framevertices = np.array([[[0, 0], frame.shape[1::-1]]], np.float32)
    #
    #
    corners, ids, rej = cv2.aruco.detectMarkers(frame,
                                                planebots.MarkerDictionary,
                                                None,
                                                None,
                                                planebots.DetectionParameters,
                                                None)
    newmtx, newroi = cv2.getOptimalNewCameraMatrix(mtx, dist, frame.shape[1::-1], 1, frame.shape[1::-1])
    # newpts = cv2.undistortPoints(framevertices, newmtx, dist)
    n = 5
    m = 6
    x, y = np.meshgrid(np.linspace(0, frame.shape[1], n), np.linspace(0, frame.shape[0], m))
    # # gr = np.ogrid()
    # gridpoints = np.array([x, y]).swapaxes(0, 2)
    # drawGridPoints(frame, gridpoints, color=(255, 0, 122))
    #
    # cv2.imwrite('detection1.png', frame)

    # newmtx, newroi = cv2.getOptimalNewCameraMatrix(mtx, dist, frame.shape[1::-1], 1, frame.shape[1::-1])
    flatpoints = np.array([[x.flatten(), y.flatten()]], np.float64).swapaxes(1, 2)
    flatpointsud = cv2.undistortPoints(flatpoints, mtx, dist, None, None, newmtx)
    # normtx = np.array(np.mat("[1.000 0 0.5; 0 1.000 .500; 0 0 1]"), np.float64)
    # pixelmtx = np.array(np.mat("[1280 0 640; 0 1024 512; 0 0 1]"), np.float64)
    # flatpointsudnorm = cv2.undistortPoints(flatpointsud, pixelmtx, None, None, None, normtx)
    # flatpointsudnormrev = cv2.undistortPoints(flatpointsudnorm, normtx, None, None, None, pixelmtx)
    gridpointsud = np.reshape(flatpointsud, (n, m, 2), order='F')

    frameud = cv2.undistort(frame, mtx, dist, None, newmtx)
    drawGridPoints(frameud, gridpointsud, color=(0, 255, 122))
    cv2.imwrite('detection2.png', frameud)

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imwrite('detection3.png', frame)

    cornerspts = list(map(lambda x: cv2.undistortPoints(x, mtx, dist, None, None, newmtx), corners))
    cv2.aruco.drawDetectedMarkers(frameud, cornerspts, ids, (0, 50, 255))
    cv2.imwrite('detection4.png', frameud)
    # vals = map(lambda x: [x[0],x[]],y)

    NODELIST = json.loads(planebots.config.get("default", "nodelist"))
    calcorners, calids, missingids = filterMarkers(cornerspts, ids, NODELIST)

    cv2.aruco.drawDetectedMarkers(frameud, calcorners, calids, (122, 50, 255))
    cv2.imwrite('detection5.png', frameud)
    centers, angles = centerMarkers(calcorners)
    for ct in centers:
        pt = tuple(map(lambda x: int(x), *ct))
        cv2.circle(frameud, pt, radius=10, color=(122, 255, 200), thickness=2)
    ref3d = getRefFromConfig(calids)
    for i, ref in enumerate(ref3d[0]):
        p0 = tuple(map(lambda x: int(x), ref))
        txt = f"{calids[i]}{p0}"
        pt = tuple(map(lambda x: int(x), *centers[i]))
        cv2.putText(frameud,
                    txt,
                    pt,
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (111, 255, 255),
                    2,
                    cv2.LINE_AA)
    cv2.imwrite('detection6.png', frameud)
    logger.info('Done')
    rv, rvec, tvec = cv2.solvePnP(ref3d, centers, mtx, dist, flags=cv2.SOLVEPNP_P3P)
    pts, jac = cv2.projectPoints(ref3d, rvec, tvec, mtx, dist)
    r3x3, _ = cv2.Rodrigues(rvec)
    Mup = np.concatenate((r3x3, tvec), 1)
    Mlo = np.mat("[0, 0, 0, 1]")
    M = np.concatenate((Mup, Mlo), 0)
    zs = np.array([0, 0, 0], np.float32).reshape((1, 1, 3))
    zs2 = np.array([550, 610, -1400], np.float32).reshape((1, 1, 3))
    answ = cv2.perspectiveTransform(zs, np.linalg.inv(M))
    answ2 = cv2.perspectiveTransform(answ, M)
    rn = np.dot(r3x3, np.mat("[0 0 1]").T)
    # for pts in
    rectangle3d = np.array(np.mat('[0 0 500; 0 1000 500; 1000 1000 500; 1000 0 500]'), np.float64)
    pts, jac = cv2.projectPoints(rectangle3d, rvec, tvec, mtx, dist)
    drawPoly(frameud, pts, closed=True, color=(0, 255, 255))
    rectangle3d = np.array(np.mat('[0 0 550; 0 1000 550; 1000 1000 550; 1000 0 550]'), np.float64)
    pts, jac = cv2.projectPoints(rectangle3d, rvec, tvec, mtx, dist)
    drawPoly(frameud, pts, closed=True, color=(0, 255, 255))
    for ptfloat in pts:
        ct = tuple(map(lambda x: int(x), *ptfloat))
        cv2.circle(frameud, ct, 20, (231, 0, 60), 3, cv2.LINE_AA)
    pos, axs = retrieveCamPos(rvec, tvec)
    txt = "pos: x:{0:04.2f} y:{1:04.2f} z:{2:04.2f}".format(*list(pos.ravel()))
    txt += "rot: x:{0:04.2f} y:{1:04.2f} z:{2:04.2f}".format(*axs.ravel())
    cv2.putText(frameud, txt, (50, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite('detection7.png', frameud)
    logger.info('Done')
    n = 5
    m = 6
    x, y = np.meshgrid(np.linspace(40, 960, n), np.linspace(40, 960, m))
    z = np.ones_like(x) * 500
    # gr = np.ogrid()
    gridpoints = np.array([x, y, z]).swapaxes(0, 2)
    flatpoints = np.array([[x.flatten(), y.flatten(), z.flatten()]], np.float64).swapaxes(1, 2).squeeze()
    pts, jac = cv2.projectPoints(flatpoints, rvec, tvec, newmtx, None)
    ptsgrid = np.reshape(pts, (n, m, 2), order='F')
    drawGridPoints(frameud, ptsgrid, drawCoords=False, drawCrosses=False)
    cv2.imwrite('detection8.png', frameud)
    dispts, jac = cv2.projectPoints(flatpoints, rvec, tvec, mtx, dist)
    ptsgrid = np.reshape(dispts, (n, m, 2), order='F')
    drawGridPoints(frame, ptsgrid, drawCoords=False, drawCrosses=False)
    cv2.imwrite('detection9.png', frame)
    logger.info("Done")
    # normalized test values:
