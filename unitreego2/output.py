from abc import ABC, abstractmethod
from typing import List

try:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    UNITREE_SDK_AVAILABLE = True
except ImportError:
    UNITREE_SDK_AVAILABLE = False
    print("WARNING: unitree_sdk2py not found. Physical robot output is disabled.")

class RobotOutput(ABC):
    @abstractmethod
    def connect(self):
        """Initialize the connection to the simulator or the real robot."""
        pass

    @abstractmethod
    def send_commands(self, target_angles: List[float], feedforward_torques: List[float]):
        """Send motor commands to the 12 joints."""
        pass

class MujocoOutput(RobotOutput):
    def __init__(self, model, data, pids, ctrl_idx):
        self.model = model
        self.data = data
        self.pids = pids
        self.ctrl_idx = ctrl_idx 
        self.dt = model.opt.timestep

    def connect(self):
        print("Connected to MuJoCo Simulator Output.")

    def send_commands(self, target_angles: List[float], feedforward_torques: List[float]):
        for leg in range(4):
            ci = self.ctrl_idx[leg]
            for j in range(3): 
                global_idx = leg * 3 + j
                target = target_angles[global_idx]
                
                # You will need to map these to however you extract qpos/qvel in your main loop
                # This is pseudocode for grabbing the current state based on your mapping
                # current_pos = self.data.qpos[...] 
                # current_vel = self.data.qvel[...] 
                
                # For now, let's assume you pass 0.0 for pos/vel just to show the structure
                # In reality, pass the actual pos/vel from self.data
                pid_torque = self.pids[leg][j].update(target, 0.0, 0.0, self.dt) 
                
                self.data.ctrl[ci[j]] = pid_torque + feedforward_torques[global_idx]

class UnitreeGo2Output(RobotOutput):
    def __init__(self, network_interface="eth0"):
        if not UNITREE_SDK_AVAILABLE:
            raise RuntimeError("Cannot initialize real robot: unitree_sdk2py is not installed.")
        
        self.interface = network_interface
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd_pub = None

    def connect(self):
        print(f"Connecting to physical Go2 via {self.interface}...")
        ChannelFactoryInitialize(0, self.interface)
        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()

    def send_commands(self, target_angles: List[float], feedforward_torques: List[float]):
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = target_angles[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            
            # Using hardware PIDs for safety and speed
            self.low_cmd.motor_cmd[i].kp = 60.0 
            self.low_cmd.motor_cmd[i].kd = 3.0  
            self.low_cmd.motor_cmd[i].tau = feedforward_torques[i]

        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.cmd_pub.Write(self.low_cmd)