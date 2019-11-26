import datetime
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

TYPES = ['allpositions', 'position', 'field', 'pngscreenshot', 'mouse', 'camera']
N_AGENTS = 10

import socket
import planebots
import json

UDP_SEND_PORT = json.loads(planebots.config.get("messaging", "udp_bridge_recv_port"))
UDP_RECV_PORT = json.loads(planebots.config.get("messaging", "udp_bridge_send_port"))
UDP_RECV_ADDRESS = planebots.config.get("messaging", "udp_bridge_addr")
UDP_SEND_ADDRESS = planebots.config.get("messaging", "udp_bridge_addr")
UDP_SEND_ADDRESS = 'localhost'
SOCK = socket.socket(socket.AF_INET,  # Internet
                     socket.SOCK_DGRAM)  # UDP
SOCK.settimeout(0.01)


def decodeHeader(bytemessage):
    try:
        message = bytemessage
        if len(message) < 16:
            raise ValueError("Incoming message to short to parse: {}<16".format(len(message)))
        logger.debug("Received data of length {}".format(len(message)))
        tsfloat = np.frombuffer(message[8:16], np.float64)[0]
        mid = np.frombuffer(message[:8], np.int64)[0]
        try:
            ts = datetime.datetime.fromtimestamp(tsfloat)
        except Exception as e:
            ts = datetime.datetime(1970, 1, 1, 0, 0, 0, 0)
        logger.debug("First bytes indicate: Timestamp: {} | MID: {} | {}".format(tsfloat, mid, ts))
        if mid == 2:
            try:
                n_agents = np.frombuffer(message[16:20], np.int32)[0]
                ids = np.frombuffer(message[24:24 + n_agents * 4], np.int32)
                logger.debug(f"Number of updates: {n_agents}: ids: {ids}")
                logger.debug("{0:<4s}|{1:<8s}|{2:<8s}|{3:<8s}|{4:<8s}".format(*"ids x y theta z".split(" ")))
                xyzth = np.frombuffer(message[24 + n_agents * 4:24 + n_agents * 4 + n_agents * 8 * 4], np.float64)
                for i in range(n_agents):
                    logger.debug(
                        "{0:<4.0f}|{1:<8.2f}|{2:<8.2f}|{3:<8.2f}|{4:<8.2f}".format(ids[i], *xyzth[i * 4:(i + 1) * 4]))
            except Exception as e:
                logger.error("Something wrong with format of package!")
                logger.debug(e)
        # try:
        #     logger.debug(TYPES[mid])
        # except Exception as e:
        #     logger.error(e)
        #     raise ValueError("Unknown MID: {}".format(mid))
        logger.debug("Sending message to connected websocket clients")
        return mid, ts
    except ValueError as e:
        logger.error(e)
        mid = -1
        ts = datetime.datetime(1970, 1, 1, 0, 0, 0, 0)
        return mid, ts


def gen1PosMessage(idx, posarray):
    mid = TYPES.index('position')
    logger.debug("Individual position update...({},{},{})".format(*posarray))
    rsc = np.array(posarray, dtype=np.float64)
    timestamp = time.time()
    ts = np.array([timestamp], np.float64).tobytes()
    msg = (mid).to_bytes(8, 'little') + ts + (idx).to_bytes(8, 'little') + rsc.tobytes()
    return msg


def genRandomPosMessage():
    mid = TYPES.index('allpositions')
    logger.debug("Generating position update...")
    rs = np.random.random((N_AGENTS, 3))
    rsc = np.multiply(rs, np.array([1000, 1000, np.pi]), dtype=np.float64)
    diag = np.diag([1, 0, 0, 0, 0])
    rsc = np.dot(diag, rsc)
    timestamp = time.time()
    ts = np.array([timestamp], np.float64).tobytes()
    msg = (mid).to_bytes(8, 'little') + ts + rsc.tobytes()
    return msg


def genRandomFieldMessage(shape):
    mid = TYPES.index('field')
    logger.debug("Generating field message with shape {}".format(shape))
    rs = np.random.random(shape)
    timestamp = time.time()
    shapebytes = np.array(shape, np.uint32).tobytes()
    ts = np.array([timestamp], np.float64).tobytes()
    msg = (mid).to_bytes(8, 'little') + ts + shapebytes + rs.tobytes()
    return msg


def sendmsg(msg, SOCK, UDP_IP, UDP_PORT):
    # tobytes = img.tobytes()
    SOCK.sendto(msg, (UDP_IP, UDP_PORT))


def genHeader(mid, timestamp=None):
    if not timestamp:
        timestamp = time.time()
    elif type(timestamp) == datetime.datetime:
        timestamp = datetime.datetime.timestamp()
    else:
        logger.error("Unable to parse timestamp!!")
    ts = np.array([timestamp], np.float64).tobytes()
    midbytes = (mid).to_bytes(8, 'little')
    msg = midbytes + ts
    return msg


def gen1PosMessage(idx, posarray):
    logger.debug("Individual position update...({},{},{})".format(*posarray))
    rsc = np.array(posarray, dtype=np.float64)
    mid = 2
    timestamp = time.time()
    ts = np.array([timestamp], np.float64).tobytes()
    msg = (mid).to_bytes(8, 'little') + ts + (idx).to_bytes(8, 'little') + rsc.tobytes()
    return msg


def sendAllPositions(positions3d, angles, ids, port=UDP_SEND_PORT):
    tpl = sorted(zip(ids, angles, positions3d), key=lambda x: x[0][0])
    ids, angles, positions3d = zip(*tpl)
    header = genHeader(2)

    msg = header + (len(ids)).to_bytes(8, 'little') + \
          np.array(ids, np.int32).tobytes()
    for i in range(len(ids)):
        msg = msg + np.array(positions3d[i], np.float64).tobytes() + \
              np.array(angles[i], np.float64).tobytes()
    logger.info(f"Sending data to: {(UDP_SEND_ADDRESS, UDP_SEND_PORT)}")
    SOCK.sendto(msg, (UDP_SEND_ADDRESS, port))


def sendMatlabPositions(positions3d, angles, ids, port=UDP_SEND_PORT):
    """ Sends all positions of max 12 markers in a fixed length packet|Header| i"""
    # send 12 positions in order
    tpl = sorted(zip(ids, angles, positions3d), key=lambda x: x[0][0])
    ids, angles, positions3d = zip(*tpl)
    ids = np.array(ids).ravel()
    header = genHeader(2)
    idx = 0
    msg = header
    for i in range(12):
        if (idx >= len(tpl)):
            msg += np.array((-i, -i, -i), np.float64).tobytes()
        elif not (int(ids[idx]) == i):
            msg += np.array((-i, -i, -i), np.float64).tobytes()
        else:
            ang = np.array(tpl[idx][1], np.float64)
            # ang = np.array(i,np.float64)
            msg += ang.tobytes()
            pos = np.array(tpl[idx][2], np.float64).ravel()[:2]
            # pos = -np.array((idx,0),np.float64)
            msg += pos.tobytes()
            idx += 1
    bytelen = len(msg) / 8
    logger.info(f"Sending data to: {(UDP_SEND_ADDRESS, port)}")
    SOCK.sendto(msg, (UDP_SEND_ADDRESS, port))
    return msg


def sendAgentObjectPositions(agents, port=UDP_SEND_PORT, TO_MM=True):
    # send 12 positions in order
    ids = [[s.mid] for s in agents]
    angles = [s.states[2] for s in agents]
    scale = not TO_MM or 1000
    positions3d = [np.array([s.states[0] * scale, s.states[1] * scale, 0], np.float64) for s in agents]
    msg = sendAllPositions(positions3d, angles, ids, port)
    return msg


def sendAgentObjectPositionsMatlab(agents, port=UDP_SEND_PORT, TO_MM=True):
    # send 12 positions in order
    ids = [[s.mid] for s in agents]
    angles = [s.states[2] for s in agents]
    scale = not TO_MM or 1000
    positions3d = [np.array([s.states[0] * scale, s.states[1] * scale, 0], np.float64) for s in agents]
    msg = sendMatlabPositions(positions3d, angles, ids, port)
    return msg


def sendLeftRight(markerid, vleft, vright, address, port):
    header = genHeader(25)

    msg = header + (1).to_bytes(8, 'little') + \
          np.array([markerid, vleft, vright], np.int32).tobytes()
    logger.info(f"Sending data to: {(address, port)}")

    d_agents = np.frombuffer(msg[16:20], np.int32)[0]
    d_vs = np.frombuffer(msg[20:], np.int32)
    SOCK.sendto(msg, (address, port))


def sendCameraPos(position, orientation):
    header = genHeader(5)
    msg = header + position.tobytes() + orientation.tobytes()
    # logger.debug()
    logger.info(f"Sending data to: {(UDP_SEND_ADDRESS, UDP_SEND_PORT)}")
    SOCK.sendto(msg, (UDP_SEND_ADDRESS, UDP_SEND_PORT))


if __name__ == '__main__':
    pass
