import logging

from pyueye import ueye

# Module to retrieve camera frames from the Ueye camera
logger = logging.getLogger(__name__)
import numpy as np
from planebots import config


class ImageBuffer:
    def __init__(self):
        self.mem_ptr = ueye.c_mem_p()
        self.mem_id = ueye.int()


bytes_per_pixel = 1
channels = 1  # 3: channels for color mode(RGB); take 1 channel for monochrome
m_nColorMode = ueye.INT()  # Y8/RGB16/RGB24/REG32
pitch = ueye.INT()
rectAOI = rect_aoi = ueye.IS_RECT()
MemID = ueye.int()
pcImageMemory = ueye.c_mem_p()
bpp = ueye.IS_CM_MONO8
buffCurrent = ImageBuffer()
buffLast = ImageBuffer()
buffs = []
numBuffers = 10
mBuff = ImageBuffer()
timeOutMS = 100
width = 640
height = 512
load_parameters_EEPROM = config["ueye"].getboolean("load_parameters_EEPROM")
camera_online = ueye.HIDS(0)


def cameraInit(load_parameters_EEPROM=load_parameters_EEPROM):
    """Initializes the camera with the correct parameters"""
    global mBuff, bpp, pitch, channels, bytes_per_pixel, tt
    hCam = ueye.HIDS(0)  # 0: first available camera;  1-254: The camera with the specified camera ID
    sInfo = ueye.SENSORINFO()
    cInfo = ueye.CAMINFO()
    # Starts the driver and establishes the connection to the camera
    nRet = ueye.is_InitCamera(hCam, None)

    logger.debug("Setting Camera Data")
    if nRet != ueye.IS_SUCCESS:
        logger.info("is_InitCamera ERROR, camera not connected?")
        return -1, 0
    # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
    nRet = ueye.is_GetCameraInfo(hCam, cInfo)
    if nRet != ueye.IS_SUCCESS:
        logger.info("is_GetCameraInfo ERROR")
    # You can query additional information about the sensor type used in the camera
    nRet = ueye.is_GetSensorInfo(hCam, sInfo)
    if nRet != ueye.IS_SUCCESS:
        logger.info("is_GetSensorInfo ERROR")
    nRet = ueye.is_ResetToDefault(hCam)
    if nRet != ueye.IS_SUCCESS:
        logger.info("is_ResetToDefault ERROR")
    # Set display mode to DIB
    nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
    if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
        # for color camera models use RGB32 mode
        logger.info("Setting colormode to black and white")
        m_nColorMode = ueye.IS_CM_MONO8
        nBitsPerPixel = ueye.INT(8)
        bytes_per_pixel = int(nBitsPerPixel / 8)
        logger.info(f"IS_COLORMODE_MONOCHROME: ")
        logger.info(f"\tm_nColorMode: \t\t{m_nColorMode}")
        logger.info(f"\tnBitsPerPixel: \t\t{nBitsPerPixel}")
        logger.info(f"\tbytes_per_pixel: \t\t {bytes_per_pixel}")

    if load_parameters_EEPROM:
        logger.debug("Loading parameters from EEPROM")
        nullint = ueye._value_cast(0, ueye.ctypes.c_uint)
        rvv = ueye.is_ParameterSet(hCam, ueye.IS_PARAMETERSET_CMD_LOAD_EEPROM, nullint, nullint)

    # Can be used to set the size and position of an "area of interest"(AOI) within an image
    nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
    if nRet != ueye.IS_SUCCESS:
        logger.error("is_AOI ERROR")
    width = rectAOI.s32Width
    height = rectAOI.s32Height

    # Prints out some information about the camera and the sensor
    logger.info(f"Camera model:\t\t {sInfo.strSensorName.decode('utf-8')}")
    logger.info(f"Camera serial no.:\t {cInfo.SerNo.decode('utf-8')}")
    logger.info(f"Maximum image width:\t {width}")
    logger.info(f"Maximum image height:\t {height}")

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
    nRet = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
    if nRet != ueye.IS_SUCCESS:
        logger.info("is_AllocImageMem ERROR")
    else:
        # Makes the specified image memory the active memory
        nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
        if nRet != ueye.IS_SUCCESS:
            logger.info("is_SetImageMem ERROR")
        else:
            # Set the desired color mode
            nRet = ueye.is_SetColorMode(hCam, m_nColorMode)

    # Activates the camera's live video mode (free run mode)
    nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
    if nRet != ueye.IS_SUCCESS:
        logger.info("is_CaptureVideo ERROR")

    # Enables the queue mode for existing image memory sequences
    nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
    if nRet != ueye.IS_SUCCESS:
        logger.info("is_InquireImageMem ERROR")
    else:
        logger.info("Press q to leave the programm")

    # ---------------------------------------------------------------------------------------------------------------------------------------

    # shutter =  int.from_bytes(sInfo.bGlobShutter.value, byteorder='big')
    # rvv = ueye.is_ParameterSet(hCam,ueye.IS_PARAMETERSET_CMD_LOAD_EEPROM,nullint,nullint)
    # Continuous image display

    tt = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
    width = rect_aoi.s32Width.value
    height = rect_aoi.s32Height.value

    for i in range(numBuffers):
        buff = ImageBuffer()
        ueye.is_AllocImageMem(hCam,
                              width, height, bpp,
                              buff.mem_ptr, buff.mem_id)

        ueye.is_AddToSequence(hCam, buff.mem_ptr, buff.mem_id)
        buffs.append(buff)
        rvIQ = ueye.is_InitImageQueue(hCam, 0)

    nRet = ueye.IS_SUCCESS

    ret = ueye.is_WaitForNextImage(hCam,
                                   100,
                                   mBuff.mem_ptr,
                                   mBuff.mem_id)
    rr = ueye.is_GetActSeqBuf(hCam, buffCurrent.mem_id, buffLast.mem_ptr, buffLast.mem_ptr)
    if (not ret):
        # cnt+=1
        array = ueye.get_data(mBuff.mem_ptr, width, height, bpp, pitch, copy=True)
        ueye.is_UnlockSeqBuf(hCam, mBuff.mem_id, mBuff.mem_ptr)

    return nRet, hCam


def cameraNewFrame(frame, hCam):
    """Retrieve a new frame from the camera"""
    # constants
    ret = 1
    cnt = 0
    while ret and cnt <= 100:
        cnt += 1
        ret = ueye.is_WaitForNextImage(hCam,
                                       timeOutMS,
                                       mBuff.mem_ptr,
                                       mBuff.mem_id)
        rr = ueye.is_GetActSeqBuf(hCam, buffCurrent.mem_id, buffLast.mem_ptr, buffLast.mem_ptr)
    if (not ret):
        logger.debug(f"ret = {ret}, copying data over to numpy array")
        fwidth, fheight = frame.shape[1::-1]

        array = ueye.get_data(mBuff.mem_ptr, fwidth, fheight, bpp, pitch, copy=True)
        arrayrs = np.reshape(array, (len(array) // fwidth, fwidth))
        cwidth, cheight = arrayrs.shape[1::-1]
        ueye.is_UnlockSeqBuf(hCam, mBuff.mem_id, mBuff.mem_ptr)

        # bytes_per_pixel = int(nBitsPerPixel / 8)
        # ...reshape it in an numpy array...
        # Fill existing buffer with new data
        # frame = np.reshape(array,(height, width, bytes_per_pixel))

        # frame[:] = np.reshape(array,(height, width))
        frame[:fheight, :fwidth] = arrayrs[:fheight, :fwidth]

        return ret, frame
    else:
        logger.error("Reading error with new frame ")
        return ret, frame


def setPixelClock(mhz, hCam=0):
    px_old = ueye.c_uint(0)
    ueye.is_PixelClock(hCam, ueye.IS_PIXELCLOCK_CMD_GET, px_old, ueye.sizeof(px_old))
    px = ueye.c_uint(mhz)
    rv = ueye.is_PixelClock(hCam, ueye.IS_PIXELCLOCK_CMD_SET, px, ueye.sizeof(px))


def setExposure(ms, hCam=0):
    ms_old = ueye.c_double(0)
    rv = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, ms_old, ueye.sizeof(ms_old))
    ems = ueye.c_double(ms)
    rv = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ems, ueye.sizeof(ems))
    ms_old = ueye.c_double(0)
    rv = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, ms_old, ueye.sizeof(ms_old))
    return rv


def setFPS(fps, hCam=0):
    ms_new = ueye.c_double(fps)
    new = ueye.c_double(0)
    rv = ueye.is_SetFrameRate(hCam, ms_new, new)
    return new


if __name__ == '__main__':
    nRet, hCam = cameraInit()
    frame = np.zeros((width, height), np.uint8)
    if nRet != -1:
        while True:
            rv = 1
            while rv:
                rv, frame = cameraNewFrame(frame, hCam)
            logger.debug("retrieved new frame")
