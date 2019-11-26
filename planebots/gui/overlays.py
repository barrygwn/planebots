import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)
import os
import json
from planebots import calibration_gigueye, MarkerDictionary, DetectionParameters, config
from planebots.vision import detection
import planebots

calibration_marker_ids = json.loads(planebots.config.get("detection", "calibration_marker_ids"))
show_grid = planebots.config.getboolean("overlays", "show_grid")
show_domain = planebots.config.getboolean("overlays", "show_domain")
show_markers = planebots.config.getboolean("overlays", "show_markers")
show_rejected_markers = planebots.config.getboolean("overlays", "show_rejected_markers")
show_positions = planebots.config.getboolean("overlays", "show_positions")
field_rotation_org = cv2.imread(os.path.join(planebots.packdir, "img", "fields", "rotational.png"))
# field_rotation = -cv2.resize(field_rotation_org,tuple(detection.field_size_mm)) +255
field_rotation = np.flipud(-field_rotation_org + 255)
field_rotation = -field_rotation_org + 255

mouse_position = (0, 0)


def mouse_callback(event, x, y, flags, param):
    """OpenCV mouse callback function"""
    # global MOUSE_POS_TUPLE
    # global IMG, FIELDFLOAT
    global mouse_position
    # global CAP
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_position = (x, y)


def rotationalfield(x, y, offset_norm=np.array([0, 0]), scaler=10, radius=0.5, x_end=1, y_end=1):
    """rotational field around a circle d=0.5 domain 0,1 -- 0,1 can be called with both
        scalars and numpy arrays as input. """
    xn = x / x_end
    yn = y / y_end
    xn = 3 * (xn - radius) - offset_norm[0]
    yn = 3 * (yn - radius) - offset_norm[1]
    ZU = (1 - xn ** 2 - yn ** 2) * xn + yn
    ZV = (1 - xn ** 2 - yn ** 2) * yn - xn
    ZUs = ZU * scaler
    ZVs = ZV * scaler
    return np.array([ZU, ZV])
    # return np.array([-1, -1])


def saveFieldImage(X, Y, ZU, ZV, savename, dpi=200, noticks=True):
    # Plots the field
    import matplotlib.pyplot as plt
    sz = np.size(ZU)
    # X,Y = np.meshgrid((ZU.shape[1])+0.5,range(ZU.shape[0])+0.5)

    fig, ax = plt.subplots(figsize=(10, 10))
    speed = np.sqrt(ZU * ZU + ZV * ZV)
    lw = 10 * speed / speed.max()
    # ax.quiver(X,Y,ZU,ZV)
    # Since the y-axis points down Y is negative
    ax.streamplot(X, Y, ZU, ZV, linewidth=lw, density=[1, 1], color='k')
    if noticks:
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        # set the axis to fit the meshgrid indices
        ax.axis([np.min(X), np.max(X), np.min(Y), np.max(Y)])
        # ax.set_aspect('equal')
        plt.savefig(os.path.join(planebots.packdir, "img", "fields", savename), dpi=dpi, bbox_inches='tight',
                    pad_inches=0)
    else:
        plt.savefig(os.path.join(planebots.packdir, "img", "fields", savename), dpi=dpi)
    # plt.show()


# def linearfield(x, y, offset_norm=np.array([0, 0]), scaler=10, radius=0.5,x_end = 1,y_end=1):
#     """rotational field around a circle d=0.5 domain 0,1 -- 0,1"""
#     xn = x/x_end
#     yn= y/y_end
#     xn = 3 * (xn - radius) - offset_norm[0]
#     yn = 3 * (yn - radius) - offset_norm[1]
#     ZU = (1 - xn ** 2 - yn ** 2) * xn + yn
#     ZV = (1 - xn ** 2 - yn ** 2) * yn - xn
#     ZUs = ZU* scaler
#     ZVs = ZV* scaler
#     return np.array([ZU, ZV])

def draw_field(npsize, offset=np.array([0, 0]), field=rotationalfield, numRows=10, numCols=10):
    """Generate an overlay of a potentialfield, potentialfield is in the domain [0,1] in R^2"""
    offset = np.array(offset, dtype=np.float32)
    vfield = np.zeros(npsize, np.uint8)
    white = 255
    for i in range(numRows):
        for j in range(numCols):
            x0 = np.float32(i + 0.5) * npsize[0] / numRows
            y0 = np.float32(j + 0.5) * npsize[1] / numCols
            xfield = x0 / npsize[0]
            yfield = y0 / npsize[1]
            dcoords = field(xfield, yfield, offset / np.array(npsize))
            dx, dy = dcoords * 2
            # x0 += offset[0]
            # y0 += offset[1]
            cv2.arrowedLine(vfield, (int(x0), int(y0)), (int(x0 + dx), int(y0 + dy)), white, 1, cv2.LINE_AA)
    return vfield


def get_field_overlay(mtx, dist, rvec, tvec, framesize, fieldsize_mm, field=rotationalfield, numRows=10, numCols=10):
    """Generates a projected field project with the current camera pose. Could be used as overlay"""
    vfield = np.zeros(framesize[1::-1], np.uint8)
    overlayColor = 255
    for i in np.linspace(0, numRows, numRows):
        for j in np.linspace(0, numCols, numCols):
            x0 = np.float32(i) * fieldsize_mm[0] / numRows
            y0 = np.float32(j) * fieldsize_mm[1] / numCols
            xfield = x0 / fieldsize_mm[0]
            yfield = y0 / fieldsize_mm[1]
            dcoords = field(xfield, yfield)
            dx, dy = dcoords * 20
            coords3d = np.array([[x0, y0, 0], [x0 + dx, y0 + dy, 0]], np.float64)
            pts, jac = cv2.projectPoints(coords3d, rvec, tvec, mtx, dist)
            p1 = tuple(map(lambda x: int(x), pts[0].ravel()))
            p2 = tuple(map(lambda x: int(x), pts[1].ravel()))
            cv2.arrowedLine(vfield, p1, p2, overlayColor, 1, cv2.LINE_AA)
    return vfield


def drawDomain(frame, rvec, tvec, mtx, dist, domain=detection.field_size_mm):
    xp, yp = list(zip([0, 0], domain))
    x, y, z = np.meshgrid(xp, yp, 0)
    domain3d = list(zip(x.ravel(), y.ravel(), z.ravel()))
    domain2d = list(zip(x.ravel(), y.ravel()))
    domain_vertices3d = np.array(domain3d, np.float64)
    # feed in newmtx and none if you want to plot in the undetected case
    newpoints, jac = cv2.projectPoints(domain_vertices3d,
                                       rvec,
                                       tvec,
                                       mtx,
                                       dist)
    detection.drawGridPoints(frame, np.reshape(newpoints, (2, 2, 2), order='F'), color=(255, 0, 0), drawLines=False)

    domain_vertices3d[2, :], domain_vertices3d[3, :] = domain_vertices3d[3, :], domain_vertices3d[2, :].copy()
    domain_vertices3d_ip = detection.interpolatePoints(domain_vertices3d.reshape(4, 3), 4)

    newpoints, jac = cv2.projectPoints(domain_vertices3d_ip,
                                       rvec,
                                       tvec,
                                       mtx,
                                       dist)
    newpointsclip = np.clip(newpoints, [0, 0], frame.shape[1::-1])
    detection.drawPoly(frame, np.squeeze(newpointsclip), color=(122, 0, 0), closed=True)


def domainToPixels(domain, rvec, tvec, mtx, dist, origin=[0, 0]):
    # rearrange in correct format
    xp, yp = list(zip(origin, domain))
    x, y, z = np.meshgrid(xp, yp, 0)
    domain3d = list(zip(x.ravel(), y.ravel(), z.ravel()))
    domain2d = list(zip(x.ravel(), y.ravel()))

    domain_vertices3d = np.array(domain3d, np.float64)

    newpoints, jac = cv2.projectPoints(domain_vertices3d, rvec, tvec, mtx, dist)
    return newpoints


def show_position(agent, frame, rvec, tvec, mtxrs, dist, newmtx, undistorted=True):
    """ Show the position of an agent"""
    x = agent.states[0]
    y = agent.states[1]

    theta = agent.states[2]
    pts = np.array([[x, y, 0], [x - 40 * np.sin(theta), y + 40 * np.cos(theta), 0]])

    if undistorted:
        newpoints, jac = cv2.projectPoints(pts, rvec, tvec, newmtx, None)
    else:
        newpoints, jac = cv2.projectPoints(pts, rvec, tvec, mtxrs, dist)

    pt2 = (int(x - 10 * np.sin(theta)), int(y + 10 * np.cos(theta)))

    ptint = np.array(newpoints, np.uint16)
    p1 = ptint[0].ravel()
    p2 = ptint[1].ravel()
    cv2.putText(frame, f'{agent.vleft:+03.0f} {agent.vright:+03.0f}', tuple(p1 + [-15, -15]), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 255))
    cv2.drawMarker(frame, tuple(p1), (255, 0, 0), cv2.MARKER_TILTED_CROSS)
    cv2.arrowedLine(frame, tuple(p1), tuple(p2), (255, 0, 0))


def rectifyCameraPose(frame_in, frame_out, rvec, tvec, mtx, domain, border_mm=[0, 0], border_pix=[0, 0]):
    # Retrieve the cornerpoints of the domain in pixel coordinates:
    border_mm = np.array(border_mm)
    newpoints = domainToPixels(domain + border_mm, rvec, tvec, mtx, None, -border_mm)
    logger.debug("generating the four corner points of the output frame")

    xp, yp = list(zip(border_pix, np.array(frame_out.shape[1::-1]) - border_pix))
    # Make a grid with 4 corner points in 3d
    x, y, z = np.meshgrid(xp, yp, 0)
    frame_vertices3d = np.array(list(zip(x.ravel(), y.ravel(), z.ravel())), np.float64)
    # Find the transformation matrix for the transformation
    H, _ = cv2.findHomography(newpoints, frame_vertices3d)
    cv2.warpPerspective(frame_in, np.array(H, np.float32), frame_out.shape[1::-1], frame_out)
    return frame_out


def mapToPnp(frame_in, frame_out, rvec, tvec, mtx, dist, domain, inverse=False):
    """Maps frame_in to frame_out, given the distortion parameters
    inverse: When inverse is selected, the frame_in has an area that is mapped to the whole frame_out
    """
    newpoints = domainToPixels(domain, rvec, tvec, mtx, dist)

    logger.debug("generating the four corner points of the png")
    xp, yp = list(zip([0, 0], frame_in.shape[1::-1]))
    # Make a grid with 4 corner points in 3d
    x, y, z = np.meshgrid(xp, yp, 0)
    frame_vertices3d = np.array(list(zip(x.ravel(), y.ravel(), z.ravel())), np.float64)
    # Find the transformation matrix for the transformation
    H, _ = cv2.findHomography(frame_vertices3d, newpoints)
    logger.debug("")
    overFrame = frame_out.copy()
    cv2.warpPerspective(frame_in, np.array(H, np.float32), frame_out.shape[1::-1], overFrame)
    cv2.add(frame_out, overFrame)


def drawRotation(showframe, rvec, tvec, mtx, dist, domain, invert=False):
    if dist is not None:
        newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, showframe.shape[1::-1], 1, showframe.shape[1::-1])
        newpoints = domainToPixels(domain, rvec, tvec, newmtx, None)

        logger.debug("generating the four corner points of the png")
        xp, yp = list(zip([0, 0], field_rotation.shape[1::-1]))
        x, y, z = np.meshgrid(xp, yp, 0)
        frame_vertices3d = np.array(list(zip(x.ravel(), y.ravel(), z.ravel())), np.float64)
        H, _ = cv2.findHomography(frame_vertices3d, newpoints)
        logger.debug("")
        overFrame = showframe.copy()
        cv2.warpPerspective(field_rotation, np.array(H, np.float32), showframe.shape[1::-1], overFrame)
        udFrame = showframe.copy()
        cv2.undistort(overFrame, newmtx, -dist, udFrame, mtx)

        overFrame = udFrame.copy()

        # newmtx,roi = cv2.getOptimalNewCameraMatrix(mtx,dist,showframe.shape[1::-1],1,showframe.shape[1::-1])
        # newmtx,roi = cv2.getOptimalNewCameraMatrix(newmtx,-dist,showframe.shape[1::-1],1,showframe.shape[1::-1])
        # udFrame = showframe.copy()
        # udFrame = cv2.undistort(overFrame,mtx,-dist,udFrame,newmtx)
    else:
        newpoints = domainToPixels(domain, rvec, tvec, mtx, dist)

        logger.debug("generating the four corner points of the png")
        xp, yp = list(zip([0, 0], field_rotation.shape[1::-1]))
        x, y, z = np.meshgrid(xp, yp, 0)
        frame_vertices3d = np.array(list(zip(x.ravel(), y.ravel(), z.ravel())), np.float64)
        H, _ = cv2.findHomography(frame_vertices3d, newpoints)
        logger.debug("")
        overFrame = showframe.copy()
        cv2.warpPerspective(field_rotation, np.array(H, np.float32), showframe.shape[1::-1], overFrame)

    # cv2.add(showframe, udFrame, showframe)

    if invert:
        cv2.subtract(showframe, overFrame, showframe)
    else:
        cv2.add(showframe, overFrame, showframe)


from planebots import coverage as cc


def drawCoverage(showframe, rvec, tvec, mtx, dist, domain, viewnr=1, invert=False):
    viewnr = viewnr % 5
    if viewnr == 0:
        logger.info("Showing coverage field")
        field_coverage_bw = fcb = np.array(np.clip(cc.Z / cc.z_star, 0, 1) * 252, np.uint8) + 3
        field_coverage_bw1 = cv2.resize(field_coverage_bw, (500, 500), interpolation=cv2.INTER_NEAREST)
        # field_coverage_bw2 = fcb = np.array(np.random.randint(3,255,(30,30)),np.uint8)
        field_coverage = np.zeros([*field_coverage_bw1.shape, 3], np.uint8)
        field_coverage[:, :, 1] = field_coverage_bw1
        field_coverage[:, :, 2] = 255 - field_coverage_bw1
        field_coverage[:10, :10, 1] = 10
    elif viewnr == 1:
        logger.info("Showing Improvement field")

        field_coverage_bw = fcb = np.array(np.clip(cc.M_impr, 0, 1) * 252, np.uint8) + 3
        field_coverage_bw1 = cv2.resize(field_coverage_bw, (500, 500), interpolation=cv2.INTER_NEAREST)
        # field_coverage_bw2 = fcb = np.array(np.random.randint(3,255,(30,30)),np.uint8)
        field_coverage = np.zeros([*field_coverage_bw1.shape, 3], np.uint8)
        field_coverage[:, :, 1] = field_coverage_bw1
        field_coverage[:, :, 2] = 255 - field_coverage_bw1
        field_coverage[:10, :10, 1] = 10
    elif viewnr == 2:
        logger.info("Showing k-field")
        field_coverage_bw = fcb = np.array(np.clip(cc.k / np.max(cc.k), 0, 1) * 252, np.uint8) + 3
        field_coverage_bw1 = cv2.resize(field_coverage_bw, (500, 500), interpolation=cv2.INTER_NEAREST)
        # field_coverage_bw2 = fcb = np.array(np.random.randint(3,255,(30,30)),np.uint8)
        field_coverage = np.zeros([*field_coverage_bw1.shape, 3], np.uint8)
        field_coverage[:, :, 1] = field_coverage_bw1
        field_coverage[:, :, 2] = 255 - field_coverage_bw1
        field_coverage[:, :, 0] = field_coverage_bw1 // 2
    elif viewnr == 3:
        logger.info("Showing F speed field")
        field_coverage_bw = fcb = np.array(np.clip(cc.Ftot / np.max(cc.Ftot), 0, 1) * 252, np.uint8) + 3
        field_coverage_bw1 = cv2.resize(field_coverage_bw, (500, 500), interpolation=cv2.INTER_NEAREST)
        # field_coverage_bw2 = fcb = np.array(np.random.randint(3,255,(30,30)),np.uint8)
        field_coverage = np.zeros([*field_coverage_bw1.shape, 3], np.uint8)
        field_coverage[:, :, 1] = field_coverage_bw1
        field_coverage[:, :, 2] = 255 - field_coverage_bw1
        field_coverage[:, :, 0] = field_coverage_bw1 // 2
    else:
        logger.info("Showing FMM grid")

        field_coverage_bw = fcb = np.array(np.clip(cc.FMMcombined, 0, 1) * 252, np.uint8) + 3
        field_coverage_bw1 = cv2.resize(field_coverage_bw, (500, 500), interpolation=cv2.INTER_NEAREST)
        # field_coverage_bw2 = fcb = np.array(np.random.randint(3,255,(30,30)),np.uint8)
        field_coverage = np.zeros([*field_coverage_bw1.shape, 3], np.uint8)
        field_coverage[:, :, 1] = field_coverage_bw1
        field_coverage[:, :, 2] = 255 - field_coverage_bw1
        field_coverage[:10, :10, 1] = 10

    if dist is not None:
        newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, showframe.shape[1::-1], 1, showframe.shape[1::-1])
        newpoints = domainToPixels(domain, rvec, tvec, newmtx, None)

        logger.debug("generating the four corner points of the image")
        xp, yp = list(zip([0, 0], field_coverage.shape[1::-1]))
        x, y, z = np.meshgrid(xp, yp, 0)
        frame_vertices3d = np.array(list(zip(x.ravel(), y.ravel(), z.ravel())), np.float64)
        H, _ = cv2.findHomography(frame_vertices3d, newpoints)
        logger.debug("")
        overFrame = showframe.copy()
        cv2.warpPerspective(field_coverage, np.array(H, np.float32), showframe.shape[1::-1], overFrame)
        udFrame = showframe.copy()
        cv2.undistort(overFrame, newmtx, -dist, udFrame, mtx)
        overFrame = udFrame.copy()

    else:
        newpoints = domainToPixels(domain, rvec, tvec, mtx, dist)

        logger.debug("generating the four corner points of the png")
        xp, yp = list(zip([0, 0], field_coverage.shape[1::-1]))
        x, y, z = np.meshgrid(xp, yp, 0)
        frame_vertices3d = np.array(list(zip(x.ravel(), y.ravel(), z.ravel())), np.float64)
        H, _ = cv2.findHomography(frame_vertices3d, newpoints)
        logger.debug("")
        overFrame = showframe.copy()
        cv2.warpPerspective(field_coverage, np.array(H, np.float32), showframe.shape[1::-1], overFrame)

    if invert:
        cv2.subtract(showframe, overFrame, showframe)
    else:
        cv2.add(showframe, overFrame, showframe)


if __name__ == '__main__':
    dst = np.zeros((200, 200))
    logger.info("Save a png of the field")
    x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    ZU, ZV = rotationalfield(x, y)
    # Since the y-axis points down Y is negative
    saveFieldImage(x, -y, ZU, -ZV, 'rotational_ticks.png', noticks=False)
    # saveFieldImage(x,y,np.zeros_like(x),np.ones_like(y),'rotational_axes.png',noticks=False)
    logger.info("Save a png of the field")
    still = cv2.imread(os.path.join("img", "still.bmp"))
    logger.debug("Loading saved pnp and matrices")
    loaded = np.load(os.path.join('settings', "still_detection.npz"))
    # Load variables:
    calibration_marker_ids, still_resolution, ref3d, mtxstill, mtx, dist, roi, newmtxstill, CamPos, CamAxis, rvec, tvec = [
        loaded[x] for x in loaded.files]
    # Make data structures for saving:
    field = still.copy()
    fieldUndistorted = still.copy()
    fieldPnP = still.copy()
    fieldPnPOverlay = still.copy()
    fieldPnPDistOverlay = still.copy()

    logger.info("Calculating map from ")
    map1, map2 = cv2.initUndistortRectifyMap(mtxstill, dist, np.eye(3), newmtxstill, still.shape[1::-1], cv2.CV_32F)
    # streamplot1500x1500 --> resize to fit
    # 400x400 -> domain
    # distorted pix (raw) to undistort
    # cv2.warpPerspective()
    cv2.remap(still, map1, map2, cv2.INTER_LINEAR, fieldUndistorted)
    # Get the cornerpoints from domain
    domain = [400, 400]
    xp, yp = list(zip([0, 0], domain))
    x, y, z = np.meshgrid(xp, yp, 0)
    domain3d = list(zip(x.ravel(), y.ravel(), z.ravel()))
    domain2d = list(zip(x.ravel(), y.ravel()))
    domain_vertices3d = np.array(domain3d, np.float64)
    newpoints_ud, jac = cv2.projectPoints(domain_vertices3d, rvec, tvec, newmtxstill, None)
    detection.drawGridPoints(fieldUndistorted, np.reshape(newpoints_ud, (2, 2, 2), order='F'), color=(255, 0, 0))
    cv2.imwrite(os.path.join("img", "overlay", "fieldUndistorted.bmp"), fieldUndistorted)
    logger.debug("Drawing grid points")
    newpoints, jac = cv2.projectPoints(domain_vertices3d, rvec, tvec, mtxstill, dist)
    detection.drawGridPoints(field, np.reshape(newpoints, (2, 2, 2), order='F'), color=(255, 255, 0))
    cv2.imwrite(os.path.join("img", "overlay", "field.bmp"), field)
    # Reproject points:
    framefield = -cv2.resize(cv2.imread("rotational.png"), tuple((400, 400))) + 255
    # Corner Points
    xp, yp = list(zip([0, 0], framefield.shape))
    x, y, z = np.meshgrid(xp, yp, 0)
    pts3d = list(zip(x.ravel(), y.ravel(), z.ravel()))
    pts = list(zip(x.ravel(), y.ravel()))
    frame_vertices3d = np.array(pts3d, np.float32)
    H, _ = cv2.findHomography(frame_vertices3d, newpoints_ud)
    cv2.warpPerspective(framefield, np.array(H, np.float32), fieldPnP.shape[1::-1], fieldPnP)
    cv2.imwrite(os.path.join("img", "overlay", "fieldPnP.bmp"), fieldPnP)
    field_color = (255, 0, 255)
    for i in range(3):
        idces = np.nonzero(fieldPnP)
        fieldUndistorted[idces[0], idces[1], :] = np.array(field_color, np.uint8)

    # field= np.bitwise_or(field,fieldPnP)
    cv2.imwrite(os.path.join("img", "overlay", "fieldUndistorted.bmp"), fieldUndistorted)

    fieldPnPDistOverlay = cv2.undistort(fieldPnP, newmtxstill, -dist, None, mtxstill)
    # for i in range(3):
    # idces = np.nonzero(fieldPnPDistOverlay)
    # field[idces[0],idces[1],:] = np.array(field_color,np.uint8)

    fo = np.add(fieldPnPDistOverlay, np.array(field, np.uint16))

    cv2.imwrite(os.path.join("img", "overlay", "fieldPnPDistOverlay.bmp"), fo)
    # cv2.addWeighted()

    # rv, rvec, tvec = cv2.solvePnP(frame_vertices3d, np.array([[620,0],[0,512],[620,512],[0,0]],np.float32).reshape((4,1,2)),newmtxstill,None, flags=cv2.SOLVEPNP_P3P) # The magic happens here!

    # cv2.projectPoints([132],)

    warpframe = still.copy()
    warpframe = cv2.warpPerspective(framefield, np.array(H, np.float32), warpframe.shape[1::-1])

    logger.info("Finding the zero rvecs for the reverse mapping of distortion")
    c3d = np.array([[620, 0, 0], [0, 512, 0], [620, 512, 0], [0, 0, 0]], np.float32).reshape((4, 1, 3))
    rv, rvec0, tvec0 = cv2.solvePnP(c3d,
                                    np.array([[620, 0], [0, 512], [620, 512], [0, 0]], np.float32).reshape((4, 1, 2)),
                                    newmtxstill, None, flags=cv2.SOLVEPNP_P3P)  # The magic happens here!
    upper_line = np.array([[0, 0, 0], [150, 0, 0], [300, 0, 0], [450, 0, 0], [600, 0, 0]], np.float64)
    prp, _ = cv2.projectPoints(upper_line, rvec0, tvec0, mtxstill, dist)
    cv2.imwrite("wf.png", warpframe)
    from planebots import detection

    frame = still.copy()
    n = 400
    x, y = pixcoords = np.meshgrid(np.linspace(0, 400, n), np.linspace(0, 400, n))
    p3d = np.zeros((n * n, 3), np.float32)
    p3d[:, 0] = x.ravel()
    p3d[:, 1] = y.ravel()
    pp3d, jac = cv2.projectPoints(p3d, rvec, tvec, mtxstill, dist)
    map12 = np.array(np.reshape(pp3d, (n, n, 2)), np.float32)
    map1b = map12[:, :, 1]
    map2b = map12[:, :, 0]

    # newframe = np.ones((400,400))
    # cv2.remap(still,map1b,map2b,cv2.INTER_LINEAR,newframe)
    # cv2.remap(still, map1 * 2, map2 * 2, cv2.INTER_LINEAR, newframe)
    # cv2.imwrite("reremap.png", newframe)
    # detection.drawGridPoints(frame,np.reshape(pp3d, (n,n, 2), order='F'))
    frame_vertices3d = np.array([[400, 0, 0], [0, 400, 0], [400, 400, 0], [0, 0, 0]], np.float64)
    # rv, rvec, tvec = cv2.solvePnP(frame_vertices3d, np.array([[620,0],[0,512],[620,512],[0,0]],np.float32).reshape((4,1,2)),newmtxstill,None, flags=cv2.SOLVEPNP_P3P) # The magic happens here!
    newpoints, jac = cv2.projectPoints(frame_vertices3d, rvec, tvec, mtxstill, dist)
    # detection.drawGridPoints(frame,np.reshape(newpoints, (2,2, 2), order='F'))
    cv2.imwrite("frame.png", frame)
    newframe1 = still.copy()

    # upper left corner points:
    # x, y = np.meshgrid(np.linspace(0, 100, 11), np.linspace(0, 100, 11))
    z = np.ones_like(x) * -0
    # flatgridpoints = np.array([[x.flatten(), y.flatten(), z.flatten()]], np.float64).swapaxes(1, 2).squeeze()

    # cameraGridPoints, jac = cv2.projectPoints(flatgridpoints, rvec, tvec, mtxstill, dist)
    # #Reshape for the grid drawing function:
    # cameraGridPoints_draw = np.reshape(cameraGridPoints, (n, m, 2), order='F')
    #
    # cameraGridPoints_ud, jac = cv2.projectPoints(flatgridpoints, rvec, tvec, newmtx, None)
    # cameraGridPoints_ud_draw = np.reshape(cameraGridPoints_ud, (n, m, 2), order='F')

    newframe2 = cv2.undistort(still, mtxstill, -dist, None, newmtxstill)
    cv2.imwrite("newframe2.png", newframe2)
    # cv2.remap(still, map2, map1, cv2.INTER_LINEAR, newframe)
    # cv2.imwrite("reremap.png",newframe)
    vfield = draw_field(dst.shape, (0, 0), rotationalfield)
    cv2.imwrite("field.png", vfield)
    still = cv2.imread(os.path.join("tools", "still.bmp"))

    mtx, dist, _, _ = calibration_gigueye
    mtxrs, roi = cv2.getOptimalNewCameraMatrix(mtx, 0, (1280, 1024), 1, still.shape[1::-1])
    NODELIST = json.loads(config.get("video_processing", "calibration_marker_ids"))
    ref3d = p3dlist = json.loads(config.get("video_processing", "calibration_markers_mm"))
    field_selected = json.loads(config.get("ueye", "field_selected"))
    field_overlays = json.loads(config.get("ueye", "field_overlays"))
    # recalculate mtx for displaying all image pixels in the ud frame:

    newmtx, roi = cv2.getOptimalNewCameraMatrix(mtxrs, dist, still.shape[1::-1], 1, still.shape[1::-1])
    corners, ids, rej = cv2.aruco.detectMarkers(still,
                                                MarkerDictionary,
                                                None,
                                                None,
                                                DetectionParameters,
                                                None)

    pnpSucces, rvec, tvec = detection.findPnP(still, mtxrs, dist, corners, ids, NODELIST, ref3d)
    overlay = get_field_overlay(mtxrs, dist, rvec, tvec, still.shape[1::-1], fieldsize_mm=(500, 500),
                                field=rotationalfield, numRows=10, numCols=10)

    cv2.imwrite("overlay.png", overlay)
    for i in range(3):
        still[:, :, i] = cv2.bitwise_or(still[:, :, i], overlay)
    cv2.imwrite("combined.bmp", still)
    logger.debug("pass")
