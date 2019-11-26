import logging
import os
import unittest

import cv2
from teamcity import is_running_under_teamcity
from teamcity.unittestpy import TeamcityTestRunner

import planebots

logger = logging.getLogger(__name__)
logger.addHandler(planebots.log_long)
logger.setLevel(logging.DEBUG)


class TestUndistort(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.modulelogger = logging.getLogger("detection")
        cls.modulelogger.handlers = [planebots.log_long]
        packdir = os.path.dirname(os.path.abspath(__package__))
        tfdir = os.path.join(packdir, "tests", "testfiles")

        cls.detpar = planebots.DetectionParameters

    def test_Detection(self):
        videoname = os.path.abspath(os.path.join('..', 'img', "sample_video.mp4"))

        logger.info(f"Opening {videoname}")
        stream = cv2.VideoCapture(videoname)
        ret, frame = stream.read()
        cv2.namedWindow("vfile")
        corners, ids, rej = cv2.aruco.detectMarkers(frame, planebots.MarkerDictionary, None, None, self.detpar)
        cv2.imshow("vfile", frame)
        cv2.waitKey(1)
        logger.debug("lalal")

        logger.warning("No logging?")

    @staticmethod
    def getTestFrame():
        videoname = os.path.abspath(os.path.join('..', 'img', "sample_video.mp4"))
        logger.info(f"Opening {videoname}")
        stream = cv2.VideoCapture(videoname)
        ret, frame = stream.read()
        return ret, frame

    def test_det_norm(self):
        ret, frame = self.getTestFrame()
        # planebots.detection.DetectMarkers(frame, )


if __name__ == '__main__':

    if is_running_under_teamcity():
        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)
    # TestUndistort.test_Detection()
