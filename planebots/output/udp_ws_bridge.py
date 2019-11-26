#!/usr/bin/env python3
# WS server example
import asyncio
import datetime
import json
import logging
import time

import numpy as np
import websockets

import planebots
from planebots.output import messages

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
bridge_debug = planebots.config["messaging"].getboolean("bridge_debug")

import os

t_start = time.time()
CNT = 0
Z_FIELD_DIMS = (20, 20)

IMAGE = np.random.randint(0, 255, (100, 100, 1), np.uint8)
USERS = set()
# Holds all states for sending to web interface 10 agents are assumed (1 to 10):
POS_0 = np.multiply(np.random.random((messages.N_AGENTS, 3)), np.array([1000, 1000, np.pi]), dtype=np.float64)
STATES = {"field": np.random.random(Z_FIELD_DIMS),
          "positions": POS_0}
calPositions = json.loads(planebots.config.get("video_processing", "calibration_markers_mm"))
calIds = json.loads(planebots.config.get("video_processing", "calibration_marker_ids"))
WSS_RECV_PORT = json.loads(planebots.config.get("messaging", "wss_bridge_send_port"))
UDP_RECV_PORT = json.loads(planebots.config.get("messaging", "udp_bridge_recv_port"))

UDP_SEND_PORT = json.loads(planebots.config.get("messaging", "udp_bridge_send_port"))
WEBSOCKET_INTERVAL = json.loads(planebots.config.get("messaging", "websocket_interval"))

# Adresses should be accessable from other clients:
WSS_RECV_ADDR = '0.0.0.0'
UDP_RECV_ADDR = '0.0.0.0'
MESSAGES = []

"""Proxies messages asynchroneously between a websocket and udp."""


def reset_messages():
    global MESSAGES
    logger.debug("Resetting all messages")
    MESSAGES = []


def register(websocket):
    logger.debug("Adding browser connection from {}:{}".format(*websocket.remote_address))
    USERS.add(websocket)


def unregister(websocket):
    logger.debug("Removing browser connection from {}:{}".format(*websocket.remote_address))
    USERS.remove(websocket)


async def notify_users(data):
    logger.debug("Sending a message to all users.. ({})".format(len(USERS)))
    if USERS:  # asyncio.wait doesn't accept an empty list
        await asyncio.wait([user.send(data) for user in USERS])


async def replynewmap(websocket, path):
    register(websocket)
    try:
        async for message in websocket:
            global CNT
            CNT += 1
            CNT = CNT % 256
            logger.debug("Received ws message... Total: {}".format(CNT))
            mid, ts = messages.decodeHeader(message, websocket.remote_address[0])
            if mid == 3:
                try:
                    now = datetime.datetime.now()
                    nowstr = now.strftime("%y%m%d%H%M%S")
                    with open(os.path.join('img', "pngfile{}.png".format(nowstr)), 'wb') as f:
                        read_data = f.write(message[16:])
                    # f.closed
                    logger.info("Screenshot saved.")
                    resolution = int(message)
                except Exception as e:
                    resolution = 100
                    print(e)
            # await websocket.send(tobytes)
    finally:
        unregister(websocket)


class EchoServerProtocol(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        towebsocket = data
        logger.debug("---Incoming UDP---")
        mid, ts = messages.decodeHeader(data, addr)
        logger.debug(f"Data Length: {len(data)}")
        hexmsg = ""
        for i in range(len(data) // 16 + 1):
            hexmsg += "|".join([f"0x{idx:02x}" for idx in data[16 * i:16 * (i + 1)]]) + '\n'
        logger.debug(hexmsg)
        logger.debug("--End Incoming UDP --")
        MESSAGES.append((data, addr))
        if len(MESSAGES):
            logger.debug(f"There are already {len(MESSAGES)} updates waiting to be sent...")
        # loop = asyncio.get_event_loop()
        # loop.create_task(self.handle_income_packet(towebsocket, addr))

    async def handle_income_packet(self, data, addr):
        await notify_users(data)
        mid, ts = messages.decodeHeader(data, addr)
        print("async update sent.. ts:{} (to {} users before {})".format(ts, len(USERS), datetime.datetime.now()))
        # loop.sock_sendall(sock, data)
        # self.transport.sendto(data, addr)


async def ws_send_loop(sleepsec):
    while 1:
        if len(MESSAGES):
            msg, addr = MESSAGES[-1]
            mid, ts = messages.decodeHeader(msg, addr)
            reset_messages()
            await notify_users(msg)
            print("async update sent.. ts:{} (to {} users before {})".format(ts, len(USERS), datetime.datetime.now()))
        await asyncio.sleep(sleepsec)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()

    if bridge_debug:
        logger.setLevel(logging.DEBUG)
        loop.set_debug(True)
        messagelog = logging.getLogger("messaging")
        messagelog.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        logger.info("Debugging disabled")
    logger.info("Starting Websocket server on {}:{}".format(WSS_RECV_ADDR, WSS_RECV_PORT))
    start_server = websockets.serve(replynewmap, WSS_RECV_ADDR, WSS_RECV_PORT, max_size=10 ** 7)
    # One protocol instance will be created to serve all client requests
    logger.info("Starting udp server {}:{}".format(UDP_RECV_ADDR, UDP_RECV_PORT))
    listen = loop.create_datagram_endpoint(
        EchoServerProtocol,
        local_addr=(UDP_RECV_ADDR, UDP_RECV_PORT),
    )

    task1 = loop.create_task(ws_send_loop(WEBSOCKET_INTERVAL))
    logger.info(f"Sending newest massage towards websocket every {WEBSOCKET_INTERVAL} seconds")
    loop.create_task(listen)
    loop.run_until_complete(start_server)
    loop.run_forever()
