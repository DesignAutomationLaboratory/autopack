import base64
import os
import pathlib
import subprocess
from typing import Any

import msgpack
import numpy as np
import zmq

from .. import __version__, logger

LUALIB_PATH = pathlib.Path(__file__).parent / "lualib" / "?.lua"
CALL_TEMPLATE = f"""
    package.path = package.path .. ';{LUALIB_PATH.absolute().as_posix()}'
    local autopack = require("autopack")

    local function runFunc()
        %s
    end

    local function packFunc(success, result)
        return autopack.pack({{
            success = success,
            result = result,
        }})
    end

    local function errHandler(err)
        return {{
            error = err,
            traceback = debug.traceback(),
        }}
    end

    local runSuccess, runResult = xpcall(runFunc, errHandler)
    local packSuccess, packResult = xpcall(packFunc, errHandler, runSuccess, runResult)

    if packSuccess then
        return packResult
    else
        return packFunc(packSuccess, packResult)
    end
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


class IPSError(RuntimeError):
    pass


class IPSInstance:
    def __init__(self, ips_path=None, port=24768):
        if ips_path is None:
            ips_path = get_ips_path()
        self.ips_path = ips_path
        self.port = port
        self.process = None
        self.socket = None
        self.version = None

    def start(self):
        subprocess.run(["taskkill", "/F", "/IM", "IPS.exe"])
        logger.info(f"Starting IPS at {self.ips_path}")
        self.process = subprocess.Popen(
            ["IPS.exe", "-port", str(self.port)],
            cwd=self.ips_path,
            env={
                **os.environ,
                "RLM_LICENSE_NAMES": "ipsm01,ipsm02,ipsm04,ipsm12,ipsm05,ipsm28",
            },
            shell=True,
        )
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://127.0.0.1:{self.port}")
        self.version = self.eval(
            f'print("Connected to Autopack v{__version__} on port {self.port}"); return Ips.getIPSVersion()'
        )
        logger.info(f"Connected to IPS {self.version} on port {self.port}")

    def _eval(self, command: str) -> bytes:
        """
        Evaluates a raw lua script in IPS and returns the raw result.

        You should probably not use this directly, use `eval` instead.
        """
        logger.debug("Evaluating IPS script: \n{0}", command)
        self._wait_socket(zmq.POLLOUT)
        self.socket.send_string(command, flags=zmq.NOBLOCK)

        self._wait_socket(zmq.POLLIN)
        msg = self.socket.recv(flags=zmq.NOBLOCK)

        return msg.strip(b"\n").strip(b'"')

    def eval(self, command: str) -> Any:
        """
        Evaluates a lua script in IPS, packs the result and returns it as a
        Python object.

        Raises an `IPSError` if the script or the packing fails.
        """
        cmd = CALL_TEMPLATE % command

        raw_response = self._eval(cmd)
        response = unpack(raw_response)

        if response["success"]:
            # void functions result in a `nil` that will not be packed
            return response.get("result", None)
        else:
            message = response["result"]["error"]
            traceback = response["result"]["traceback"]

            raise IPSError(f"IPS call failed: {message}\n{traceback}")

    def kill(self):
        subprocess.run(["taskkill", "/F", "/IM", "IPS.exe"])

    def _wait_socket(self, flags, timeout=1000):
        """Wait for socket or crashed process."""
        while True:
            if self.process.poll() is not None:
                raise RuntimeError("Process died unexpectedly")
            if self.socket.poll(timeout, flags) != 0:
                return
