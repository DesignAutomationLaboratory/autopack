import base64
import os
import pathlib
import subprocess

import msgpack
import zmq

LUALIB_PATH = pathlib.Path(__file__).parent / "lualib" / "?.lua"
LOAD_LUALIB_SCRIPT = f"""
    Script.resetStateWhenFinished(false)
    package.path = package.path .. ';{LUALIB_PATH.absolute().as_posix()}'
    local msgpack = require("MessagePack")
    local base64 = require("base64")

    local function pack(data)
        return base64.encode(msgpack.pack(data))
    end
    _G.autopack = {{pack=pack}}
    _G.msgpack = msgpack
    _G.base64 = base64
"""


def get_ips_path():
    try:
        return os.environ["AUTOPACK_IPS_PATH"]
    except KeyError:
        raise Exception(
            'Please set the environment variable AUTOPACK_IPS_PATH to the path of your IPS installation, e.g. by running the following in a command prompt and then restarting: setx AUTOPACK_IPS_PATH "C:\\Program Files\\IPS\\IPS_2023-R2-SP1-HarnessRouter_v3.0_x64"'
        )


def unpack(payload):
    return msgpack.unpackb(base64.b64decode(payload))


class IPSInstance:
    def __init__(self, ips_path=None, port="24768"):
        if ips_path is None:
            ips_path = get_ips_path()
        self.ips_path = ips_path
        self.port = port
        self.socket = None

    def start(self, verify_connection=True, load_libs=True):
        subprocess.run(["taskkill", "/F", "/IM", "IPS.exe"])
        subprocess.Popen(
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
            assert (
                response == b'"ping"\n'
            ), f"IPS did not respond as expected: {response}"
            self.call(
                f"print('Connection to Autopack on port {self.port} successfully verified!')"
            )

        if load_libs:
            self.call(LOAD_LUALIB_SCRIPT)

    def call(self, command):
        self.socket.send_string(command)
        msg = self.socket.recv()
        return msg

    def call_unpack(self, command):
        raw_response = self.call(command)
        stripped_response = raw_response.strip(b"\n").strip(b'"')
        return unpack(stripped_response)

    def kill(self):
        subprocess.run(["taskkill", "/F", "/IM", "IPS.exe"])
