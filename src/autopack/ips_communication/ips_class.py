import zmq
import os
import subprocess

class IPSInstance:
    def __init__(self, ips_path, port="24768"):
        self.ips_path = ips_path
        self.port = port
        self.socket = None
    def start(self):
        subprocess.run(['taskkill', '/F', '/IM', 'IPS.exe'])
        subprocess.Popen(["IPS.exe", "-port", self.port], cwd=self.ips_path, env={**os.environ,"RLM_LICENSE_NAMES": "ipsm01,ipsm02,ipsm04,ipsm12,ipsm05,ipsm28"}, shell=True)
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://127.0.0.1:" + self.port)
    def call(self, command):
        self.socket.send_string(command)
        msg = self.socket.recv()
        return msg
    def kill(self):
        subprocess.run(['taskkill', '/F', '/IM', 'IPS.exe'])

