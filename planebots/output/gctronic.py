import ctypes
# import planebots
import logging
import os
import sys
import time

# Load DLL into memory.
logger = logging.getLogger(__name__)
import planebots
import json
import numpy as np

elisa_numbers = json.loads(planebots.config.get("gctronic", "elisa_numbers"))
elisa_ids = json.loads(planebots.config.get("gctronic", "elisa_ids"))
elisa3_frequency_hz = json.loads(planebots.config.get("gctronic", "elisa3_frequency_hz"))
b = 0.0408  # Diameter of the robots
k = 211  # Conversion from vlr to bits


def vtou(vau, omega, s=6000):
    """measurement function of the elisa robot"""

    # k=6.1
    # 0.6ms*v == 127 --> k = 211
    # k = 211
    vl = (vau + (1 / 2) * omega * b)
    vr = (vau - (1 / 2) * omega * b)
    vlbit = int(np.clip(vl * k, -127, 127))
    vrbit = int(np.clip(vr * k, -127, 127))
    return [vrbit, vlbit]


def utov(lr):
    """Converts elisa byte inputs values to velocities and rotation in m/s and rad/s"""
    l, r = lr
    # b = 0.0408  # Wheelbase according to the data sheet
    # k = 211  # to conver to m/s 127 ^= 0.6m/s
    u_vau, u_omega = np.array([0.5 * (l + r), (r - l) / b]) / k  # wheelbase of 40.8mm
    return [u_vau, u_omega]


def toElisa(Comm, agents):
    for agent in agents:
        Comm.setLeftSpeed(agent.number, agent.vleft)
        Comm.setRightSpeed(agent.number, agent.vright)


def toElisaLoop(Comm, agents, interval):
    tic = time.perf_counter()
    while True:
        toc = time.perf_counter()
        elapsed = toc - tic
        toElisa(Comm, agents)
        sleeptime = interval - elapsed
        time.sleep(max(sleeptime, 0.0))
        tic = time.perf_counter()

    exit()


# defines a wrapper class to the Elisa3 robots:
class Elisa3Comm(object):
    # Install libusb drivers:
    # cd sudo apt-get install libusb-1.0-0-dev
    # Compile with libusb included:
    # gcc -c ../usb-comm.c -lusb ../elisa3-lib.c -lusb -fPIC
    # Compile a static library .a file
    # bundle .o file to a dynamically linked file for use in python
    # gcc -shared elisa3-lib.o -lusb-1.0 usb-comm.o -lusb-1.0 -o elisa3.so
    # Copy elisa3.so to this directory
    # To avoid needing root privilege:
    # -list devices with lsusb: Bus 001 Device 006: ID 1915:0101 Nordic Semiconductor ASA
    # Add rule to	/etc/udev/rules.d to avoid root elevation
    # I.e. touch /etc/udev/rules.d 50-elisa3.rules
    # SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTR{idProduct}=="0101", MODE="0666"
    # Re-login
    if sys.platform == 'win32':
        # Compile the elisa3 dll for your platform with static linkage, so all functions can be found.
        # dllpath = os.path.join("D:\\data\\thesis\elisa3\Elisa-3 demo\pc-side-elisa3-library\\bin\Release","libpc-side-elisa3-library.dll")
        libpath = os.path.join("libpc-side-elisa3-library.dll")
        library = ctypes.cdll.LoadLibrary(libpath)
    else:
        libpath = os.path.join(planebots.packdir, 'lib', "elisa3.so")
        # libpath = os.path.join("lib","elisa3.so")
        logger.debug(f"Opening libpath {libpath}")
        # libpath = os.path.join("./","lib")
        library = ctypes.cdll.LoadLibrary(libpath)
        # raise NotImplementedError

    def __init__(self, AddressList=[3655, 3658, 3533], suppressError=True):
        self.nRobots = len(AddressList)
        self.AddressList = AddressList
        self.cList = x = (ctypes.c_int32 * self.nRobots)()
        for i in range(len(AddressList)):
            self.cList[i] = AddressList[i]
        self.AddrPt = ctypes.cast(self.cList, ctypes.POINTER(ctypes.c_int64))
        self.add_functions()
        try:
            logger.info("Starting communications with Elisa3 Dongle")
            self.library.startCommunication(self.AddrPt, self.nRobots)
            for i in range(self.nRobots):
                logger.info(f"Address:{AddressList[i]} Id:{self.getIdFromAddress(AddressList[i])}")
        except OSError as e:
            logger.error(e)
            logger.error("Is the dongle connected? Could not open..")
            if suppressError:
                pass
            else:
                raise e
            # raise ConnectionError(e)
            # exit()

    def add_functions(self):
        # Functions working with implicit type conversion:
        self.computeVerticalAngle = self.library.computeVerticalAngle
        self.getIdFromAddress = self.library.getIdFromAddress
        self.setLeftSpeed = self.library.setLeftSpeed
        self.setRightSpeed = self.library.setRightSpeed
        self.setBlue = self.library.setBlue
        self.setRed = self.library.setRed
        self.setGreen = self.library.setGreen
        self.getBatteryAdc = self.library.getBatteryPercent
        # unsigned char waitForUpdate(int robotAddr, unsigned long us);
        self.waitForUpdate = self.library.waitForUpdate

    def close(self):
        commrv = self.library.stopCommunication()


def getOdometry(Comm, Address):
    # ddx is accelerating, ddy anticlockwise turning
    accXYZ = np.array([Comm.library.getAccX(Address), Comm.library.getAccY(Address), Comm.library.getAccZ(Address)],
                      np.float32)
    odoXYTHETA = np.array(
        [Comm.library.getOdomXpos(Address), Comm.library.getOdomYpos(Address), Comm.library.getOdomTheta(Address)],
        np.float32)
    odoXYTHETA[2] = odoXYTHETA[2] * np.pi / 180 / 10
    return accXYZ, odoXYTHETA


if __name__ == '__main__':
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    logger.info(f"VerticalAngle test:{Elisa3Comm.library.computeVerticalAngle(10,10)}")
    Comm = Elisa3Comm(AddressList=[3333, 3655, 3656, 3533])
    l = 5
    r = -5
    try:
        for ci in range(Comm.nRobots):
            for i in range(Comm.nRobots):
                Address = Comm.AddressList[i]
                try:
                    if i == ci:
                        # Set the blue LED intensity to the level of the right trigger:
                        # intensity = int(gamepad_states["ABS_RZ"]*100/255)
                        Comm.setLeftSpeed(Address, l)
                        Comm.setRightSpeed(Address, r)
                        Comm.setBlue(Address, 10)
                        Comm.setRed(Address, 1)
                        ans = (ctypes.c_int32 * 36)()
                        logger.info(f"Battery at :{Comm.library.getBatteryAdc(Address)} %")
                        Comm.library.getAccX.restype = ctypes.c_int
                        a, b = getOdometry(Comm, Address)
                        logger.info(f"{a}  {b}")
                        logger.info(f"accx at :{Comm.library.getAccX(Address)} %")
                        logger.info(f"accy at :{Comm.library.getAccY(Address)} %")
                        logger.info(f"accz at :{Comm.library.getAccZ(Address)} %")
                        logger.info(f"odo at :{Comm.library.getOdomTheta(Address)} %")
                    else:
                        Comm.setLeftSpeed(Address, 0)
                        Comm.setRightSpeed(Address, 0)
                        Comm.setBlue(Address, 0)
                        Comm.setRed(Address, 0)
                        # Show the communication is live:
                        Comm.setGreen(Address, 1)
                except Exception as e:
                    logger.error(e)

            time.sleep(1)

        for i in range(Comm.nRobots):
            Comm.setLeftSpeed(Comm.AddressList[i], 0)
            Comm.setRightSpeed(Comm.AddressList[i], 0)
        #     Sleep in order to open the connection long enough to ensure the new settings arrived
        # time.sleep(0.1)
        rv = Comm.waitForUpdate(Comm.AddressList[i], 1000)
        logger.info(f"rv:{rv}")
        # Comm.close()
    except (KeyboardInterrupt, SystemExit) as e:
        pass
