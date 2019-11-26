# import colorama

import planebots.vision.detection
from planebots.vision.detection import *

# import barrygwn.genpurpose as gp
logger = logging.getLogger(__name__)
import configparser
from pyueye import ueye
import cv2
import os
import numpy as np

toc = time.clock()
# logger.addHandler(gp.console_debugger(intellij=True))
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
logger.info(__name__)
detpar = planebots.DetectionParameters
detparindex = 0

"""Script for conveniently finding detection parameters optimal for the current video. Uses ../img/sample_video.mp4 to test the marker detection"""


def adjustDetpar(frame, detpar, attr, upordown):
    # detparorg = detpar.copy()
    curval = getattr(detpar, attr)
    if upordown:
        if type(curval) == int:
            setattr(detpar, curattr, curval + 1)
        else:
            setattr(detpar, curattr, curval * 1.2)
    else:
        if type(curval) == int:
            setattr(detpar, curattr, curval - 1)
        else:
            setattr(detpar, curattr, curval / 1.2)
    try:
        corners, ids, rej = cv2.aruco.detectMarkers(frame, planebots.MarkerDictionary, None, None, detpar)
    except Exception as e:
        logger.error(e)
        setattr(detpar, curattr, curval)
    return detpar


lastkeycode = 0
if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    logger.info("Starting parameter adjuster")
    videoname = os.path.abspath(os.path.join('..', 'img', "sample_video.mp4"))
    logger.info(f"Opening {videoname}")
    stream = cv2.VideoCapture(videoname)

    opened = stream.isOpened()
    mtx, dist, _, _ = planebots.calibration_gigueye
    cv2.namedWindow("vfile")
    cnt = 0

    h_cam = ueye.HIDS(0)
    ret = ueye.is_InitCamera(h_cam, None)

    if ret != ueye.IS_SUCCESS:
        pass


    def capture_video(self, wait=False):
        wait_param = ueye.IS_WAIT if wait else ueye.IS_DONT_WAIT
        return ueye.is_CaptureVideo(self.h_cam, wait_param)


    # import ids

    while 1:
        try:
            if cnt < 50:
                ret, frameor = stream.read()
            cnt += 1
        except AttributeError as e:
            logger.error(e)
            stream = cv2.VideoCapture(videoname)
            continue
        # time.sleep(1)
        # newMtx = cv2 np.zeros((3,3),np.float32)
        # if FAST:
        frame = frameor.copy()

        # Test if only with distorted points the corners can be reconstructed:
        tic = time.clock()
        corners, ids, rej = cv2.aruco.detectMarkers(frameor, planebots.MarkerDictionary, None, None, detpar)
        toc = time.clock()

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.aruco.drawDetectedMarkers(frame, rej, None, (122, 0, 122))
        halfsize = (int(frame.shape[1] / 1), int(frame.shape[0] / 1))
        rsz = cv2.resize(frame, halfsize)
        blank = np.ones_like(rsz) * 5
        rsz = np.hstack((blank[:, :300, :], rsz))
        lines = [f"({k:30},{getattr(detpar,k)})" for k in planebots.DetectionKeys]

        addTextlines(rsz, planebots.DetectionKeys, offset=(80, 0))

        vals = [f"{getattr(detpar,k):>6.3f}" for k in planebots.DetectionKeys]
        vals[detparindex] = vals[detparindex] + "<-"
        vals.append(f"{toc*1000-tic*1000:04.2f}ms")
        vals.append(lastkeycode)
        addTextlines(rsz, vals)
        cv2.putText(rsz, "Press [s] and [w] to toggle through the list", (0, 425), 0, .8, (255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(rsz, "Press [-] and [+] to toggle change values", (0, 450), 0, .8, (255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(rsz, "Press [d] to save to detpar.ini", (0, 475), 0, .8, (255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(rsz, "Press [q] quit", (0, 500), 0, .8, (255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)
        cv2.imshow("vfile", rsz)
        key = cv2.waitKeyEx(1) & 0xFFFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            cv2.VideoCapture.release()
            raise SystemExit
        elif key == ord('+'):
            curattr = planebots.DetectionKeys[detparindex]
            curval = getattr(detpar, curattr)
            detpar = adjustDetpar(frame, detpar, curattr, 1)
        elif key == ord('-'):
            curattr = planebots.DetectionKeys[detparindex]
            curval = getattr(detpar, curattr)
            detpar = adjustDetpar(frame, detpar, curattr, 0)
        elif key == ord('s'):
            detparindex += 1
            detparindex %= planebots.DetectionKeys.__len__()
        elif key == ord('w'):
            detparindex += planebots.DetectionKeys.__len__() - 1
            detparindex %= planebots.DetectionKeys.__len__()

        elif key == ord('d'):
            config = configparser.ConfigParser()
            config.add_section('detectionparameters')
            for k in planebots.DetectionKeys:
                config['detectionparameters'][k] = f'{getattr(detpar,k)}'
            with open('detpar.ini', 'w+') as file:
                config.write(file)
        if key != 0xFFFF:
            try:
                lastkeycode = f"0x{key:02X}:{key}:{np.array([key],np.uint16 ).tobytes().decode('utf-8')}"
            except UnicodeDecodeError as e:
                pass
