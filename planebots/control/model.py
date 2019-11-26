import logging

import cv2
import numpy as np

import planebots

logger = logging.getLogger(__name__)
from planebots.output import messages
from planebots.vision import detection
import time

import planebots.control.controllers as controllers
from planebots.control.observers import ukFilter


#


# Submodule implementing a class for each agent with several convenience functions.


class Agent():
    """Class for creating an object representing an agents. Functions are defined mainly te draw sprites in the gui,
    parse incoming commands, save odometry data, and updating the filter with measurement data. """

    @staticmethod
    def parseStringInput(commandString):
        """Parses commandstrings in the format to be parsed by the Miabot agents"""
        try:
            # str.
            vleft = 0
            vright = 0
            if commandString.count("[") == 1 and commandString.count("]") == 1:
                if commandString.find("s") == 1:
                    # fetched [s]
                    return 0, 0, 0
                mid = 0
                velocitypart = commandString
            else:
                # First parse mid
                pclosing = commandString.index(']')
                idstring = commandString[commandString.index('[') + 1:pclosing]
                mid = int(idstring)
                velocitypart = commandString[pclosing + 1:]
            pclosing = velocitypart.index(']')
            vpart = velocitypart[velocitypart.index('[') + 1:pclosing]
            vclosing = vpart.index('>')
            leftstring = vpart[:vclosing].replace("<", "").replace(">", "").replace(",", "")
            rightstring = vpart[vclosing:].replace("<", "").replace(">", "").replace(",", "")
            if rightstring.find("r") != -1:
                vright = int(rightstring.replace("r", "").replace("l", ""))
            elif rightstring.find("l") != -1:
                vleft = int(rightstring.replace("r", "").replace("l", ""))
            if leftstring.find("r") != -1:
                vright = int(leftstring.replace("r", "").replace("l", ""))
            elif leftstring.find("l") != -1:
                vleft = int(leftstring.replace("r", "").replace("l", ""))

            # str.find()
            return mid, vleft, vright
        except Exception as e:
            logger.error(e)

    def __init__(self, states=[0, 0, 0, 0, 0], mid=0, number=0000, scale=1):
        self.states = list(states)
        self.mid = mid
        # elisa3 number
        self.number = number
        self.scale = scale
        self.vleft = 0
        self.vright = 0
        self.odometry = np.array([0., 0., 0.])  # odometry data from onboard
        self.odometry0 = np.array([0., 0., 0.])
        self.imu = np.array([0., 0., 0.])  # imu data from onboard
        self.visual = np.array([0., 0., 0.])  # data from cam
        self.timeOdometry = time.perf_counter()
        self.timeplanebots = time.perf_counter()
        self.percent = -1
        # Measurement data:
        self.z_vis = np.array([0., 0., 0.])  # data from cam
        self.z_tsv = time.perf_counter()
        self.z_tsvdt = 0.1
        self.z_imu = np.array([0., 0])
        self.z_odom = np.array([0., 0])
        self.z_odom_last = np.array([0., 0])
        self.z_xyth = np.array([0., 0, 0])
        self.z_ts = 0
        self.z_tsv = 0
        self.z_dt = 0.1
        self.z_uw = [0, 0]
        self.z_count = 0
        # self.z_xhat = np.array([0., 0, 0, 0, 0])
        self.t_0 = time.perf_counter()
        self.x_hat = np.zeros(5)
        self.ukf_v = controllers.ukf_v
        self.ukf_e = controllers.ukf_e
        self.t_last = time.perf_counter()
        self.filter = ukFilter()
        self.filter.predict(u=None, dt=0.1)  # Must be called before updates
        self.filterv = ukFilter()
        self.filtere = ukFilter()
        self.filtere.predict(u=None, dt=0.1)  # Must be called before updates

    # def updateFilterElisa(self):
    #     if self.m_battPerc > 0: #Check if the agent is connected
    #         ts = time.perf_counter()-self.t_last
    #         l,r = self.vleft,self.vright
    #         uvo = np.array([0.5 * (l + r),(r-l) / 0.0408])/170# wheelbase of 40.8mm
    #         self.ukf_e.x = self.x_hat
    #         self.ukf_e.predict(u=uvo*0,dt=ts)
    #         dr,dl = self.z_odom -self.z_odom_last
    #         tl,tr = [dl/ts/5,dr/ts/5]
    #         z = [tl,tr,self.z_imu[0]]
    #
    #         xprior = self.ukf_e.x.copy()
    #
    #         self.ukf_e.update(z,R=np.diag([0.001,0.0001,1]))
    #         xafter = self.ukf_e.x.copy()
    #         delta = xprior - xafter
    #         self.x_hat = self.ukf_e.x
    #         self.t_last+=ts
    #         self.z_odom_last = self.z_odom

    # def updateFilterplanebots(self,xyth):
    #     ts = time.perf_counter()-self.t_last
    #     l,r = self.vleft,self.vright
    #     uvo = np.array([0.5 * (l + r),(l - r) / 0.0408])/170# wheelbase of 40.8mm
    #     logger.debug("Updating location with planebots data")
    #
    #     self.z_vis = xyth
    #
    #     try:
    #         th_hat =self.x_hat[6]
    #         th_z =xyth[2]
    #         diff = th_hat - th_z
    #         if abs(diff) > np.pi:
    #             self.x_hat[6] -= np.sign(diff)*np.pi*2
    #             th_hat =self.x_hat[6]
    #             diff = th_hat - th_z
    #             logger.debug("Large difference")
    #
    #         zv = [xyth[0]/1000,xyth[1]/100,xyth[2]]
    #
    #         self.x_hat = self.ukf_v.x
    #         self.ukf_v.x = self.x_hat
    #         # self.ukf_v.predict(u=uvo,dt=ts)
    #
    #         # Predect that the agent will continue on its way:
    #         u = [np.linalg.norm([self.x_hat[1],self.x_hat[4]]), self.x_hat[7]]
    #         self.ukf_v.predict(u=u,dt=ts)
    #
    #         self.ukf_v.update(xyth/[1000,1000,1],R=np.diag([0.2,0.2,5]))
    #         # self.x_hat = self.ukf_v.x
    #         self.t_last += ts
    #     except Exception as e:
    #         logger.error("e")
    def measurementString(self):
        pass

    def getAllStates(self, Comm):
        """Gets all data available on an Elisa3"""
        Address = self.number
        m_prox = np.array([Comm.library.getProximity(Address, i) for i in range(8)], np.uint16)
        m_ground = np.array([Comm.library.getGround(Address, i) for i in range(4)], np.uint16)
        m_battRaw = Comm.library.getBatteryAdc(Address)
        m_battPerc = Comm.library.getBatteryPercent(Address)
        # odomxyz,imu  = self.getOdometry(Comm)
        m_select = Comm.library.getSelector(Address)
        m_odomlr = np.array([Comm.library.getLeftMotSteps(Address), Comm.library.getRightMotSteps(Address)], np.float32)
        # chargeState  = Comm.library.robotIsCharging(Address)
        # btnState     = Comm.library.buttonIsPressed(Address)
        m_radioQuality = Comm.library.getRFQuality(Address)

        return m_prox, m_ground, m_battRaw, m_battPerc, m_select, m_odomlr, m_radioQuality

        # color = (1,1,1)
        #         # prox -> 8x [0->1023]
        #         # getGround -> 4x [0->1023]
        #         # getBatteryPercent 0-100

    # def convertSpeeds(self):
    #     uvec = np.array([np.cos(the[i]), np.sin(the[i])])
    #     uvec2 = np.array([dtxe[i - 1], dtye[i - 1]])
    #     umag = np.linalg.norm(uvec2)
    #     cos = np.dot(uvec, uvec2)
    #     u = np.sign(cos) * umag

    def getOdometry(self, Comm):
        Address = self.number
        # ddx is accelerating, ddy anticlockwise turning
        accXYZ = np.array([Comm.library.getAccX(Address), Comm.library.getAccY(Address), Comm.library.getAccZ(Address)],
                          np.float32)
        odoXYTHETA = np.array(
            [Comm.library.getOdomXpos(Address), Comm.library.getOdomYpos(Address), Comm.library.getOdomTheta(Address)],
            np.float32)

        odoXYTHETA[0] = odoXYTHETA[0] / 1000
        odoXYTHETA[1] = odoXYTHETA[1] / 1000
        odoXYTHETA[2] = odoXYTHETA[
                            2] * np.pi / 180  # \return current orientation of the robot expressed in 1/10 of degree (3600 degrees for a full turn).
        self.percent = Comm.library.getBatteryPercent(Address)
        x0, y0, th0 = self.odometry
        self.odometry = np.array(odoXYTHETA, np.float32)  # odometry data from onboard
        x, y, th = self.odometry
        dx, dy, dth = np.array((x0, y0, th0)) - [x, y, th]
        up = np.array([np.cos(th), np.sin(th)])
        uv = np.array([dx, dy])
        umag = np.linalg.norm(uv)
        # cos = np.dot(up, uv)
        cos = self.vleft / 2 + self.vright / 2
        if umag > 1e-8:
            u = np.sign(cos) * umag
        else:
            u = 0
        w = -dth
        self.m_imu = np.array(accXYZ, np.float32)  # imu data from onboard
        self.timeOdometry = time.perf_counter()
        # logger.debug(f"Odom update: {self.number} {self.odometry }  {self.imu}")
        self.z_dt = self.dt = time.perf_counter() - self.z_ts
        self.z_ts = time.perf_counter()
        self.z_imu = accXYZ[:2]
        self.z_uw = np.array([u, w]) / self.z_dt
        self.z_u = np.array([self.vleft, self.vright])
        self.z_count += 1
        self.z_odom = np.array([Comm.library.getLeftMotSteps(Address), Comm.library.getRightMotSteps(Address)],
                               np.float32)
        self.z_xyth = self.odometry
        m_prox, m_ground, m_battRaw, m_battPerc, m_select, m_odomlr, m_radioQuality = self.getAllStates(Comm)
        self.m_prox = m_prox
        self.m_ground = m_ground
        self.m_battRaw = m_battRaw
        self.m_battPerc = m_battPerc
        self.m_select = m_select
        self.m_odomlr = m_odomlr
        self.m_radioQuality = m_radioQuality

        i = 1

        # Update the npz
        return self.odometry, self.imu

    def updateAgents(agents, centersInPlane, anglesInPlane, ids):
        """Updates the states of the agent with the position, angle and ID retrieved from the visual odometry"""
        # Extract each elisa mid
        elisaIds = [_.mid for _ in agents]
        found = 0
        for i, nr in enumerate(elisaIds):
            # Check if there is a marker with such an ID:
            a1, _, _ = detection.filterMarkers(anglesInPlane, ids, [nr])
            p1, _, _ = detection.filterMarkers(centersInPlane, ids, [nr])
            # If so update corresponding agents position and pose:
            if len(a1):
                agents[i].states[2] = agents[i].visual[2] = a1[0]
                agents[i].states[:2] = agents[i].visual[:2] = p1[0].ravel()[:2]
                agents[i].z_xyth = agents[i].states[:3]
                agents[i].lastUpdate = agents[i].timeplanebots = time.perf_counter()
                found += 1
        return agents

    @classmethod
    def fromConfig(cls):
        """ returns a list of agents as specified in the config.ini file"""
        import planebots.output.gctronic as gctronic
        # Make the agents
        agents = []
        for i in range(len(gctronic.elisa_numbers)):
            agent = cls(mid=gctronic.elisa_ids[i], states=[0, 0, 0, 0, 0], number=gctronic.elisa_numbers[i])
            logger.info(agent)
            agents.append(agent)
        return agents

    @staticmethod
    def encodeAgentInputs(agentList):
        """ Encodes the states of the agents into an UDP packet, to be send over the network."""
        header = messages.genHeader(25)
        # Header plus the length
        ln = np.array([len(agentList)], np.int32)
        msg = header + ln.tobytes()

        for agent in agentList:
            nrs = np.array([agent.mid, agent.number], np.int32)
            arr = np.array([*agent.states, agent.vleft, agent.vright], np.float64)
            encoded = nrs.tobytes() + arr.tobytes()
            decoded = np.frombuffer(encoded, np.float64)
            msg += encoded

        return msg

    @classmethod
    def decodeAgentInputs(cls, packet):
        """ Decodes the states of the agents from an UDP packet, into a list of objects."""
        agentList = []
        mid, ts = messages.decodeHeader(packet)
        ln = np.frombuffer(packet[16:20], np.int32).ravel()

        for i in range(ln[0]):
            agent = cls()

            agent.mid, agent.number = np.frombuffer(packet[64 * i + 20:64 * i + 28], np.int32)
            ss = np.frombuffer(packet[64 * i + 28:64 * i + 28 + 7 * 8], np.float64)
            agent.states = ss[:5]
            agent.vleft = ss[5]
            agent.vright = ss[6]
            agentList.append(agent)
        return agentList

    def moveinaxes(self, x, u, dt):
        """
        x
        ^
        |  theta <-
        |           \
        |            |
         - - - - - - - ->y
        :param x: state
        :param u: reference [v,theta_dot]
        :return: x new state
        """
        vau, omega = u
        X = list(x)
        Xk1 = [0, 0, 0, 0, 0]
        if np.abs(omega * dt) > 0.00001:
            logger.debug("Moving along a circular trajectory")
            R = vau / omega
            dx = R * (np.sin(X[2] + omega * dt) - np.sin(X[2]))
            dy = -R * (np.cos(X[2] + omega * dt) - np.cos(X[2]))
            Xk1[0] = X[0] + dx
            Xk1[1] = X[1] + dy
            dth = omega
            Xk1[4] = dth
        else:
            dx = vau * np.cos(X[2]) * dt
            dy = vau * np.sin(X[2]) * dt
            Xk1[0] = X[0] + dx
            Xk1[1] = X[1] + dy
            Xk1[4] = 0
        deltath = Xk1[4] * dt
        Xk1[2] = (X[2] + deltath + 2 * np.pi) % (2 * np.pi)
        Xk1[3] = vau

        return Xk1

    def axb(self, l, r, dt):
        """
        Wheel speeds are specified as a number with a fixed scaling, from 0 to approximately 2000 maximum (positive or
         negative). The actual rate in terms of pulses-per-second is speed*50, so that a speed of 1000 is actually
         50,000 pulses per second, i.e. a linear speed of approximately 2.0m/sec. (factor 500)
        :param l:
        :param r:
        :param dt:
        :return:
        """

        l = l / 1000
        r = r / 1000

        X = list(self.states)
        Xk1 = [0, 0, 0, 0, 0]
        # vau =X[3]
        b = 0.0408
        vau = 0.5 * (l + r)
        omega = (l - r) / b
        if np.abs(omega * dt) > 0.00001:
            R = vau / omega
            dx = R * (np.sin(X[2] + omega * dt) - np.sin(X[2]))
            dy = -R * (np.cos(X[2] + omega * dt) - np.cos(X[2]))
            Xk1[0] = X[0] + dx
            Xk1[1] = X[1] + dy
            dth = omega
            Xk1[4] = dth
        else:
            dx = vau * np.cos(X[2]) * dt
            dy = vau * np.sin(X[2]) * dt
            Xk1[0] = X[0] + dx
            Xk1[1] = X[1] + dy
            Xk1[4] = 0.
        deltath = Xk1[4] * dt
        Xk1[2] = (X[2] + deltath + 2 * np.pi) % (2 * np.pi)
        Xk1[3] = vau

        self.states = Xk1

    def move(self, l, r, dt):
        """
        Wheel speeds are specified as a number with a fixed scaling, from 0 to approximately 2000 maximum (positive or
         negative). The actual rate in terms of pulses-per-second is speed*50, so that a speed of 1000 is actually
         50,000 pulses per second, i.e. a linear speed of approximately 2.0m/sec. (factor 500)
        :param l:
        :param r:
        :param dt:
        :return:
        """

        l = l / 500
        r = r / 500

        X = list(self.states)
        Xk1 = [0, 0, 0, 0, 0]
        # vau =X[3]
        b = 0.1
        vau = 0.5 * (l + r)
        omega = (l - r) / b
        if np.abs(omega * dt) > 0.00001:
            R = vau / omega
            dx = R * (np.cos(X[2] + omega * dt) - np.cos(X[2]))
            dy = R * (np.sin(X[2] + omega * dt) - np.sin(X[2]))
            Xk1[0] = X[0] + dx
            Xk1[1] = X[1] + dy
            dth = omega
            Xk1[4] = dth
        else:
            dx = - vau * np.sin(X[2]) * dt
            dy = vau * np.cos(X[2]) * dt
            Xk1[0] = X[0] + dx
            Xk1[1] = X[1] + dy
            Xk1[4] = 0
        deltath = Xk1[4] * dt
        Xk1[2] = (X[2] + deltath + 2 * np.pi) % (2 * np.pi)
        Xk1[3] = vau

        self.states = Xk1

    def moveString(self, commandString, dt=.1):
        mid, left, right = Agent.parseStringInput(commandString)
        if mid == self.mid:
            self.move(left, right, dt)
        else:
            raise ValueError

    def __repr__(self):
        """Compact representation in string format of the states of the agent"""
        str1 = "ID:{5:>02.0f}|Nr{6}|{9:02d}%".format(
            *self.states, self.mid, self.number, self.vleft, self.vright, self.percent)
        odo = self.odometry0 + self.odometry
        str1 += f"|{self.z_count%100:02d}"

        str1 += f"|l:{self.vleft:02d},r:{self.vright:02d}"
        str1 += "|x_hat:x:{0:+04.3f} " \
                "y:{1:+04.3f} " \
                "θ:{2:+04.3f} " \
                "v{3:+04.2f} " \
                "O{4:+04.2f} ".format(*self.x_hat[:5] * [1, 1, 1, 1, 1])

        str1 += "|z_vis:x:{0:+04.0f}y:{1:+04.0f}θ:{2:+04.2f}".format(*self.z_vis)
        str1 += "|z_uw:u:{0:+04.3f}dθ:{1:+04.3f}".format(*self.z_uw)
        str1 += "|z_xyth:x:{0:+04.3f}y:{1:+04.3f}θ:{2:>+04.3f}".format(*self.z_xyth)
        str1 += "|z_imu:ddx:{0:+03.0f}ddy:{1:+03.0f}".format(*self.z_imu)
        str1 += "|z_odom:x:{0:+04.0f}y:{1:+04.0f}".format(*self.z_odom)
        # str1 += "x:{0:>+04.1f} y:{1:>+04.1f} θ:{2:>03.1f} v:{3:>04.1f} ω:{4:>04.1f} l{7},r{8}".format(*self.states, self.mid, self.number, self.vleft, self.vright, self.percent)
        return str1

    def show_position(self, img):
        """Draw a marker and an arrow indicating the position and pose of the agent on an image"""
        # Place sprite on position x,y
        scale = self.scale
        x = int(self.states[0] * scale)
        y = int(self.states[1] * scale)
        theta = self.states[2]
        cv2.drawMarker(img, (x, y), (122, 122, 122), cv2.MARKER_TILTED_CROSS)
        pt2 = (int(x + 10 * np.sin(theta)), int(y + 10 * np.cos(theta)))
        cv2.arrowedLine(img, (x, y), pt2, (122, 122, 122))

    def show_position_pnp(self, img, rvec, tvec, mtx, dist, showOdometry=False):
        """Draw a marker and an arrow indicating the position and pose of the agent on an image, projected using the
        camera pose"""
        # Place sprite on position x,y
        if showOdometry:
            x = self.odometry0[0] + np.cos(self.odometry0[2]) * self.odometry[0] - np.sin(self.odometry0[2]) * \
                self.odometry[1]
            y = self.odometry0[1] + np.sin(self.odometry0[2]) * self.odometry[0] + np.cos(self.odometry0[2]) * \
                self.odometry[1]
            theta = self.odometry0[2] + self.odometry[2]
            color = (0, 255, 0)
            x = self.x_hat[0] * 1000
            y = self.x_hat[1] * 1000
            theta = self.x_hat[2]
        else:
            x = self.z_vis[0]
            y = self.z_vis[1]
            theta = self.z_vis[2]
            color = (0, 122, 255)

        dOpts = {'thickness': 1}
        length = 40

        domain_vertices3d = np.array([[x, y, 0], [x + length * np.cos(theta), y + length * np.sin(theta), 0]],
                                     np.float64)
        # feed in newmtx and none if you want to plot in the undetected case
        newpoints, jac = cv2.projectPoints(domain_vertices3d,
                                           rvec,
                                           tvec,
                                           mtx,
                                           dist)

        pt1 = np.array(newpoints[0], np.int16).ravel()
        pt1 = tuple(map(lambda x: int(x), pt1))
        pt2 = np.array(newpoints[1], np.int16).ravel()
        pt2 = tuple(map(lambda x: int(x), pt2))

        cv2.drawMarker(img, pt1, color, cv2.MARKER_TILTED_CROSS, **dOpts)
        cv2.arrowedLine(img, pt1, pt2, color, **dOpts)

    def placesprite(self, img, sprite):
        """Draw a sprite indicating the position and pose of the agent on an image"""
        # Place sprite on position x,y
        scale = self.scale
        x = int(self.states[0] * scale)
        y = int(self.states[1] * scale)
        shape = sprite.shape[:2]
        delta = shape[0] // 2
        bg = img[y - delta:y + delta, x - delta:x + delta]
        if delta > y or delta > x or delta > -x + img.shape[1] or delta > -y + img.shape[0]:
            logger.info("problem?")
            # cv2.drawMarker(img,(x,y),(122,122,122),cv2.MARKER_TILTED_CROSS)
        else:
            try:
                imshape = img.shape
                isColor = (len(imshape) == 3) or False
                if isColor:
                    img[y - delta:y + delta, x - delta:x + delta] = np.bitwise_or(sprite, bg)
                else:
                    img[y - delta:y + delta, x - delta:x + delta] = np.bitwise_or(sprite[:, :, 1], bg)
            except ValueError as e:
                logger.error(e)
        cv2.drawMarker(img, (x, y), (122, 122, 122), cv2.MARKER_TILTED_CROSS)
        txtopts = {'fontFace': 1, 'fontScale': 1.3, 'color': (255, 255, 255), 'thickness': 1, 'lineType': cv2.LINE_AA}
        cv2.putText(img, f"{x},{y},{self.states[2]:04.2f}", (x, y), **txtopts)

    @staticmethod
    def sprite_rotate(sprite, angle_radians):
        """Rotate the sprite of the agent with angle_radians radians"""
        shape = sprite.shape[:2]
        center = (shape[0] // 2, shape[1] // 2)
        M = cv2.getRotationMatrix2D(center, angle_radians * 180 / np.pi, 1)
        spr = cv2.warpAffine(sprite, M, shape)
        return spr
        #     # cv2.drawMarker(img,(x,y),(122,122,122),cv2.MARKER_TILTED_CROSS)

    def sprite_gen(self, px_outer=100, px_inner=50, color=(122, 122, 0)):
        """ Generate a sprite of the marker with a color and specified dimensions. px_outer has to be at lest 2**.5 * px_inner
        to not lose parts of the marker sprite during rotations"""
        sprite = cv2.aruco.drawMarker(planebots.MarkerDictionary, self.mid, px_inner, borderBits=2)
        cv2.rectangle(sprite, (0, 0), (px_inner, px_inner), color, 2, lineType=cv2.LINE_AA)
        sprite_color = np.ones((*sprite.shape, 3), np.uint8)
        sprite_color[:, :, 0] = np.bitwise_and(sprite, np.ones_like(sprite) * color[0])
        sprite_color[:, :, 1] = np.bitwise_and(sprite, np.ones_like(sprite) * color[1])
        sprite_color[:, :, 2] = np.bitwise_and(sprite, np.ones_like(sprite) * color[2])
        spritew = np.zeros((px_outer, px_outer, 3), np.uint8)
        rmin = px_outer // 2 - px_inner // 2
        rmax = px_outer // 2 + px_inner // 2
        spritew[rmin:rmax, rmin:rmax] = sprite_color
        logger.debug("Create gray background")
        spritew += 50
        return spritew


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    mid = Agent.parseStringInput("[2][<111l>,<112r>]")
    bot = Agent(mid=2)
    bot.moveString("[2][<111l>,<112r>]")

    bot.states = [0.2, 0.2, -np.pi * 3 / 4, 0, 0]

    window = cv2.namedWindow("MODEL", cv2.WINDOW_AUTOSIZE)
    BLANK = np.zeros((500, 500, 3), np.uint8)
    sprite = bot.sprite_gen()
    while True:
        MASK = BLANK.copy()

        bot.moveString("[2][<10l>,<112r>]", dt=0.05)

        sprite_rotated = Agent.sprite_rotate(sprite, bot.states[2])
        bot.placesprite(MASK, sprite_rotated)
        logger.info("done")
        cv2.imshow("MODEL", MASK)
        cv2.waitKey(50)
