import base64
import os
import pathlib
import subprocess

import msgpack
import numpy as np
import zmq

LUALIB_PATH = pathlib.Path(__file__).parent / "lualib" / "?.lua"
LOAD_LUALIB_SCRIPT = f"""
    Script.resetStateWhenFinished(false)
    package.path = package.path .. ';{LUALIB_PATH.absolute().as_posix()}'
    _G.autopack = require('autopack')
"""


def get_ips_path():
    try:
        return os.environ["AUTOPACK_IPS_PATH"]
    except KeyError:
        raise Exception(
            'Please set the environment variable AUTOPACK_IPS_PATH to the path of your IPS installation, e.g. by running the following in a command prompt and then restarting: setx AUTOPACK_IPS_PATH "C:\\Program Files\\IPS\\IPS_2023-R2-SP1-HarnessRouter_v3.0_x64"'
        )


def encode_hook(obj, chain=None):
    """
    Hook for msgpack to pack custom objects.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj if chain is None else chain(obj)


def pack(payload):
    return base64.b64encode(msgpack.packb(payload, default=encode_hook))


def unpack(payload):
    return msgpack.unpackb(base64.b64decode(payload))


class IPSInstance:
    def __init__(self, ips_path=None, port="24768"):
        if ips_path is None:
            ips_path = get_ips_path()
        self.ips_path = ips_path
        self.port = port
        self.process = None
        self.socket = None
        self._version = None

    def start(self, verify_connection=True, load_libs=True):
        subprocess.run(["taskkill", "/F", "/IM", "IPS.exe"])
        self.process = subprocess.Popen(
            ["IPS.exe", "-port", self.port],
            cwd=self.ips_path,
            env={
                **os.environ,
                "RLM_LICENSE_NAMES": "ipsm01,ipsm02,ipsm04,ipsm12,ipsm05,ipsm28",
            },
            shell=True,
        )
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://127.0.0.1:" + self.port)

        if verify_connection:
            response = self.call('return "ping"')
            assert response == b"ping", f"IPS did not respond as expected: {response}"
            self.call(
                f"print('Connection to Autopack on port {self.port} successfully verified!')"
            )

        if load_libs:
            self.call(LOAD_LUALIB_SCRIPT)

    def call(self, command, strip=True):
        self._wait_socket(zmq.POLLOUT)
        self.socket.send_string(command, flags=zmq.NOBLOCK)

        self._wait_socket(zmq.POLLIN)
        msg = self.socket.recv(flags=zmq.NOBLOCK)

        if strip:
            return msg.strip(b"\n").strip(b'"')
        else:
            return msg

    def call_unpack(self, command):
        response = self.call(command, strip=True)
        return unpack(response)

    def kill(self):
        subprocess.run(["taskkill", "/F", "/IM", "IPS.exe"])

    def _wait_socket(self, flags, timeout=1000):
        """Wait for socket or crashed process."""
        while True:
            if self.process.poll() is not None:
                raise RuntimeError("Process died unexpectedly")
            if self.socket.poll(timeout, flags) != 0:
                return

    @property
    def version(self):
        if self._version is None:
            self._version = self.call("return Ips.getIPSVersion()").decode(
                "unicode_escape"
            )
        return self._version
