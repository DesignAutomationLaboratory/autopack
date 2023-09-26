import os
import subprocess

import zmq


def get_ips_path():
    try:
        return os.environ["AUTOPACK_IPS_PATH"]
    except KeyError:
        raise Exception(
            'Please set the environment variable AUTOPACK_IPS_PATH to the path of your IPS installation, e.g. by running the following in a command prompt and then restarting: setx AUTOPACK_IPS_PATH "C:\\Program Files\\IPS\\IPS_2023-R2-SP1-HarnessRouter_v3.0_x64"'
        )


class IPSInstance:
    def __init__(self, ips_path=None, port="24768"):
        if ips_path is None:
            ips_path = get_ips_path()
        self.ips_path = ips_path
        self.port = port
        self.socket = None

    def start(self):
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

    def call(self, command):
        self.socket.send_string(command)
        msg = self.socket.recv()
        return msg

    def kill(self):
        subprocess.run(["taskkill", "/F", "/IM", "IPS.exe"])
