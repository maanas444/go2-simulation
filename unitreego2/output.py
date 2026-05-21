from abc import ABC, abstractmethod
from typing import List
import numpy as np

try:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
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
    def get_low_state(self):
        """Return the current state of the robot (IMU, Foot Contacts, Motor states)."""
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

    def get_low_state(self):
        # Creates a dummy state so the HardwareBridge doesn't crash in Sim
        class MockIMU:
            def __init__(self, acc, gyro): 
                self.accelerometer = acc
                self.gyroscope = gyro
        class MockState:
            def __init__(self, acc, gyro): 
                self.imu_state = MockIMU(acc, gyro)
                self.foot_force = [50, 50, 50, 50] # Fake foot contacts
                self.wireless_remote = [0]*40 

        # Pull actual Z-accel from MuJoCo (Ensure these sensors exist in your XML)
        accel = self.data.sensor("imu_acc").data if "imu_acc" in self.model.sensor_names else [0, 0, 9.81]
        gyro = self.data.sensor("imu_gyro").data if "imu_gyro" in self.model.sensor_names else [0, 0, 0]
        return MockState(accel, gyro)

    def send_commands(self, target_angles: List[float], feedforward_torques: List[float]):
        for leg in range(4):
            ci = self.ctrl_idx[leg]
            for j in range(3): 
                global_idx = leg * 3 + j
                target = target_angles[global_idx]
                
                # IMPORTANT: You still need to map your current qpos/qvel here based on your indices
                current_p = 0.0 
                current_v = 0.0 
                
                pid_torque = self.pids[leg][j].update(target, current_p, current_v, self.dt) 
                self.data.ctrl[ci[j]] = pid_torque + feedforward_torques[global_idx]


class UnitreeGo2Output(RobotOutput):
    def __init__(self, network_interface="eth0", dt=0.002):
        if not UNITREE_SDK_AVAILABLE:
            raise RuntimeError("Cannot initialize real robot: unitree_sdk2py is not installed.")
        
        self.interface = network_interface
        self.dt = dt
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = LowState_()
        
        self.cmd_pub = None
        self.state_sub = None
        
        # --- Software Integral (I) Term variables ---
        self.error_integral = np.zeros(12)
        self.ki = 0.5  # Adjust this gain based on testing

    def _state_callback(self, msg: "LowState_"):
        """Automatically updates internal state whenever the robot broadcasts it."""
        self.low_state = msg

    def connect(self):
        print(f"Connecting to physical Go2 via {self.interface}...")
        ChannelFactoryInitialize(0, self.interface)
        
        # Publisher for sending commands
        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()
        
        # Subscriber for receiving IMU, Motor, and Foot data
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self._state_callback, 10)

    def get_low_state(self):
        """Exposes the robot's state to the EKF and HardwareBridge."""
        return self.low_state

    def send_commands(self, target_angles: List[float], feedforward_torques: List[float]):
        for i in range(12):
            target = target_angles[i]
            
            # --- NEW: Hybrid PID Calculation ---
            # 1. Get current angle from the latest low_state
            current = self.low_state.motor_state[i].q
            
            # 2. Accumulate integral error
            error = target - current
            self.error_integral[i] += error * self.dt
            
            # 3. Calculate the 'I' Torque
            ki_torque = self.ki * self.error_integral[i]
            
            # 4. Pack the command
            m = self.low_cmd.motor_cmd[i]
            m.q = target
            m.dq = 0.0
            
            # Hardware handles P and D
            m.kp = 60.0 
            m.kd = 3.5  
            
            # Software handles I (added to feedforward)
            m.tau = feedforward_torques[i] + ki_torque

        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.cmd_pub.Write(self.low_cmd)