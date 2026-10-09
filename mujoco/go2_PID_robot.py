import os
import sys
import time
import math
import struct
from typing import List, Optional

import numpy as np

# The Unitree SDK is commonly cloned on the robot rather than installed into
# the active virtual environment. Add the standard checkout locations before
# importing it, while allowing an explicit SDK path via UNITREE_SDK_PATH.
_SDK_PATHS = [
    os.environ.get("UNITREE_SDK_PATH"),
    "/home/unitree/unitree_sdk2_python",
    os.path.expanduser("~/unitree_sdk2_python"),
]
for _sdk_path in _SDK_PATHS:
    if _sdk_path and os.path.isdir(_sdk_path) and _sdk_path not in sys.path:
        sys.path.insert(0, _sdk_path)

try:
    import mujoco
    import mujoco.viewer
except Exception:  # pragma: no cover - optional during robot-only deployment
    mujoco = None
    mujoco_viewer = None

try:
    import pygame
except Exception:  # pragma: no cover - optional if using real robot remote only
    pygame = None

try:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    UNITREE_SDK_AVAILABLE = True
except Exception:  # pragma: no cover - depends on robot runtime
    ChannelPublisher = None
    ChannelSubscriber = None
    ChannelFactoryInitialize = None
    unitree_go_msg_dds__LowCmd_ = None
    LowCmd_ = None
    LowState_ = None
    CRC = None
    UNITREE_SDK_AVAILABLE = False


SIM_XML_PATH = os.path.expanduser('~/unitree_mujoco/unitree_robots/go2/scene.xml')
ROBOT_INTERFACE = 'eth0'

# -----------------------------------------------------------------------------
# Geometry and gait copied from the original MuJoCo controller.
# -----------------------------------------------------------------------------
L_THIGH = 0.213
L_CALF = 0.213
THIGH_LIM = (-1.571, 3.491)
CALF_LIM = (-2.723, -0.838)
HIP_LIM = (-1.047, 1.047)

REAL_STAND = {
    "FR": (0.018, 0.667, -1.377),
    "FL": (-0.018, 0.663, -1.369),
    "RR": (0.085, 0.660, -1.353),
    "RL": (-0.082, 0.658, -1.351),
}

REAL_SIT = {
    "FR": (0.061, 1.236, -2.761),
    "FL": (-0.068, 1.241, -2.770),
    "RR": (0.383, 1.243, -2.756),
    "RL": (-0.402, 1.244, -2.758),
}

HIP_STAND, THIGH_STAND, CALF_STAND = 0.0, 0.662, -1.363
FOOT_Z_STAND = -(L_THIGH * math.cos(THIGH_STAND) + L_CALF * math.cos(THIGH_STAND + CALF_STAND))

TRANSITION_DURATION = 1.5
STEP_FREQ = 2.0
STEP_HEIGHT = 0.08
STEP_LEN_X = 0.18
STEP_LEN_Y = 0.08
TURN_STRIDE = 0.10
PHASE_OFFSET = [0.0, math.pi, math.pi, 0.0]
FWD_AXIS_SIGN = -1.0

KP, KI, KD = [100.0, 250.0, 200.0], [5.0, 10.0, 10.0], [3.0, 6.0, 5.0]
TORQUE_MAX = [23.7, 23.7, 45.43]

CTRL_IDX = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
QPOS_IDX = [[10, 11, 12], [7, 8, 9], [16, 17, 18], [13, 14, 15]]
QVEL_IDX = [[9, 10, 11], [6, 7, 8], [15, 16, 17], [12, 13, 14]]

AXIS_LX, AXIS_LY, AXIS_RX = 0, 1, 3
BTN_A, BTN_X, BTN_Y, DEADZONE = 0, 2, 3, 0.12
_LEG_KEYS = ["FR", "FL", "RR", "RL"]
EKF_WARMUP_SECS = 1.5


class StateEstimator:
    def __init__(self, dt, initial_height=0.28):
        self.dt = dt
        self.x = np.array([[initial_height], [0.0]])
        self.P = np.eye(2) * 1.0
        self.Q = np.diag([0.005, 0.005])
        self.R = 0.01
        self.enabled = False

    def reset(self, initial_height):
        self.x = np.array([[initial_height], [0.0]])
        self.P = np.eye(2) * 1.0
        self.enabled = False

    def predict(self, z_accel):
        if not self.enabled:
            return
        a_world = z_accel - 9.81
        F = np.array([[1, self.dt], [0, 1]])
        B = np.array([[0.5 * self.dt**2], [self.dt]])
        self.x = F @ self.x + B * a_world
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measured_z):
        if not self.enabled:
            self.x[0, 0] = measured_z
            return
        H = np.array([[1, 0]])
        y = measured_z - (H @ self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S
        self.x = self.x + K * y
        self.P = (np.eye(2) - K @ H) @ self.P


class PIDController:
    def __init__(self, kp, ki, kd, torque_max):
        self.kp, self.ki, self.kd, self.torque_max = kp, ki, kd, torque_max
        self.integral, self.integral_max = 0.0, 10.0

    def update(self, target, pos, vel, dt):
        error = target - pos
        self.integral = np.clip(self.integral + error * dt, -self.integral_max, self.integral_max)
        torque = (self.kp * error) + (self.ki * self.integral) - (self.kd * vel)
        return float(np.clip(torque, -self.torque_max, self.torque_max))


def ik(px, pz):
    r = np.clip(math.sqrt(px * px + pz * pz), 0.05, L_THIGH + L_CALF - 0.005)
    cos_c = (L_THIGH**2 + L_CALF**2 - r**2) / (2.0 * L_THIGH * L_CALF)
    calf = -(math.pi - math.acos(np.clip(cos_c, -1.0, 1.0)))
    alpha = math.atan2(px, -pz)
    cos_b = (L_THIGH**2 + r**2 - L_CALF**2) / (2.0 * L_THIGH * r)
    thigh = alpha + math.acos(np.clip(cos_b, -1.0, 1.0))
    return float(np.clip(thigh, *THIGH_LIM)), float(np.clip(calf, *CALF_LIM))


def lerp_pose(leg_key, t):
    s = t * t * (3.0 - 2.0 * t)
    sit, stand = REAL_SIT[leg_key], REAL_STAND[leg_key]
    return tuple(sit[i] + s * (stand[i] - sit[i]) for i in range(3))


def dz(v):
    return v if abs(v) > DEADZONE else 0.0


def reset_robot(data):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[2] = 0.12
    data.qpos[3] = 1.0
    for leg in range(4):
        hip, thigh, calf = REAL_SIT[_LEG_KEYS[leg]]
        qp = QPOS_IDX[leg]
        data.qpos[qp[0]], data.qpos[qp[1]], data.qpos[qp[2]] = hip, thigh, calf


def compute_foot_target(leg, ph, stride_x, stride_y, yaw):
    key = _LEG_KEYS[leg]
    side_sign = 1.0 if leg in (0, 2) else -1.0
    hip_nominal = REAL_STAND[key][0]
    yaw_stride = yaw * TURN_STRIDE * side_sign
    total_stride_x = stride_x + yaw_stride

    if ph < math.pi:
        prog = ph / math.pi
        swing_t = math.sin(math.pi * prog)
        px = -total_stride_x / 2.0 + total_stride_x * prog
        pz = FOOT_Z_STAND + STEP_HEIGHT * swing_t
        py_delta = stride_y / 2.0 - stride_y * prog
    else:
        prog = (ph - math.pi) / math.pi
        px = total_stride_x / 2.0 - total_stride_x * prog
        pz = FOOT_Z_STAND
        py_delta = -stride_y / 2.0 + stride_y * prog

    hip_t = hip_nominal + py_delta
    thigh_t, calf_t = ik(px, pz)
    return hip_t, thigh_t, calf_t


class UnitreeRemoteHandler:
    """Decode the Unitree wireless remote payload to joystick-like values."""

    def __init__(self, deadzone=0.1):
        self.deadzone = deadzone
        self.state = {
            "lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0,
            "a": 0, "b": 0, "x": 0, "y": 0,
            "f1": 0, "start": 0,
        }

    def _apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def update(self, remote_data):
        if not remote_data or len(remote_data) < 40:
            return self.state
        b1, b2 = remote_data[2], remote_data[3]
        self.state["f1"] = (b1 >> 6) & 1
        self.state["start"] = (b1 >> 2) & 1
        self.state["a"] = (b2 >> 0) & 1
        self.state["b"] = (b2 >> 1) & 1
        self.state["x"] = (b2 >> 2) & 1
        self.state["y"] = (b2 >> 3) & 1
        try:
            sticks = struct.unpack('<5f', bytes(remote_data[4:24]))
            self.state["lx"] = self._apply_deadzone(sticks[0])
            self.state["rx"] = self._apply_deadzone(sticks[1])
            self.state["ry"] = self._apply_deadzone(sticks[2])
            self.state["ly"] = self._apply_deadzone(sticks[4])
        except struct.error:
            pass
        return self.state


class RealGo2Controller:
    def __init__(self, interface=ROBOT_INTERFACE):
        if not UNITREE_SDK_AVAILABLE:
            raise RuntimeError(
                'unitree_sdk2py is required for real robot deployment. '
                'Install the Unitree SDK and retry.'
            )
        self.interface = interface
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        # LowState_ is a required-field generated DDS message in this SDK and
        # cannot be constructed without a complete payload. It is populated
        # asynchronously by the rt/lowstate subscriber instead.
        self.low_state = None
        self.cmd_pub = None
        self.state_sub = None
        self.remote = UnitreeRemoteHandler(deadzone=0.12)
        self.error_integral = np.zeros(12)
        self.ki = 0.5
        self.motors_enabled = False

    def connect(self):
        ChannelFactoryInitialize(0, self.interface)
        self.cmd_pub = ChannelPublisher('rt/lowcmd', LowCmd_)
        self.cmd_pub.Init()
        self.state_sub = ChannelSubscriber('rt/lowstate', LowState_)
        self.state_sub.Init(self._state_callback, 10)
        print(f'Connected to real Go2 on {self.interface}')

    def _state_callback(self, msg):
        self.low_state = msg

    def wait_for_state(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while self.low_state is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if self.low_state is None:
            raise RuntimeError(
                'No rt/lowstate message received. Check the robot network and '
                f'interface {self.interface!r}.'
            )

    def enable_motors(self):
        self.motors_enabled = True
        print('WARNING: motor command output enabled.')

    def set_target_angles(self, targets: List[float], feedforward_torques: Optional[List[float]] = None):
        if feedforward_torques is None:
            feedforward_torques = [0.0] * 12
        if self.low_state is None:
            raise RuntimeError('Cannot send motor commands before receiving rt/lowstate.')
        if not self.motors_enabled:
            return
        for i in range(12):
            current = self.low_state.motor_state[i].q
            error = targets[i] - current
            self.error_integral[i] += error * 0.002
            ki_torque = self.ki * self.error_integral[i]
            m = self.low_cmd.motor_cmd[i]
            m.q = targets[i]
            m.dq = 0.0
            m.kp = 60.0
            m.kd = 3.5
            m.tau = feedforward_torques[i] + ki_torque
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.cmd_pub.Write(self.low_cmd)

    def read_remote(self):
        if self.low_state is None:
            return self.remote.state
        return self.remote.update(self.low_state.wireless_remote)


class MuJoCoGo2Controller:
    def __init__(self, xml_path=SIM_XML_PATH):
        if mujoco is None:
            raise RuntimeError('MuJoCo is not installed or unavailable.')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002
        self.pids = [[PIDController(KP[j], KI[j], KD[j], TORQUE_MAX[j]) for j in range(3)] for _ in range(4)]

    def connect(self):
        reset_robot(self.data)
        mujoco.mj_forward(self.model, self.data)
        print('Connected to MuJoCo Go2 simulator')

    def read_imu_z_accel(self):
        return self.data.sensor('imu_acc').data[2]

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def set_joint_targets(self, targets, torques=None):
        if torques is None:
            torques = [0.0] * 12
        for leg in range(4):
            qp, qv, ci = QPOS_IDX[leg], QVEL_IDX[leg], CTRL_IDX[leg]
            for j in range(3):
                idx = leg * 3 + j
                self.data.ctrl[ci[j]] = self.pids[leg][j].update(
                    targets[idx], self.data.qpos[qp[j]], self.data.qvel[qv[j]], self.model.opt.timestep
                ) + torques[idx]


def build_joystick():
    if pygame is None:
        return None
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print('No joystick found; will try robot remote control fallback.')
        return None
    joy = pygame.joystick.Joystick(0)
    joy.init()
    return joy


def get_remote_axes(remote_state):
    return remote_state['ly'], remote_state['lx'], remote_state['ry']


def main(use_real_robot=False, enable_motors=False):
    if use_real_robot:
        if not UNITREE_SDK_AVAILABLE:
            raise RuntimeError('Robot deployment requested, but unitree_sdk2py is not installed.')
        robot = RealGo2Controller(interface=ROBOT_INTERFACE)
        robot.connect()
        print('Waiting for rt/lowstate...')
        robot.wait_for_state()
        print('Received rt/lowstate.')
        if enable_motors:
            robot.enable_motors()
        else:
            print('Motor output is DISABLED. Use --enable-motors only after validating the mapping.')
        joy = None
    else:
        robot = MuJoCoGo2Controller()
        robot.connect()
        joy = build_joystick()

    ekf = StateEstimator(0.002, initial_height=0.12)
    sit_height = 0.12
    state, t_global, sit_stand_t = 'SIT', 0.0, 0.0
    ekf_warmup_elapsed = 0.0
    prev_btn = {BTN_A: False, BTN_X: False, BTN_Y: False}
    sim_time, wall_origin = 0.0, time.perf_counter()

    if not use_real_robot:
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
    else:
        viewer = None

    try:
        while True:
            if joy is not None:
                pygame.event.pump()
                cur_btn = {b: bool(joy.get_button(b)) for b in (BTN_A, BTN_X, BTN_Y)}
                just_pressed = {b: cur_btn[b] and not prev_btn[b] for b in cur_btn}
                prev_btn = dict(cur_btn)
                ly = dz(joy.get_axis(AXIS_LY))
                lx = dz(joy.get_axis(AXIS_LX))
                rx = dz(joy.get_axis(AXIS_RX))
            else:
                remote = robot.read_remote() if use_real_robot else {'lx': 0.0, 'ly': 0.0, 'ry': 0.0}
                cur_btn = {'a': remote.get('a', 0), 'x': remote.get('x', 0), 'y': remote.get('y', 0)}
                just_pressed = {'a': cur_btn['a'] and not prev_btn.get('a', False),
                               'x': cur_btn['x'] and not prev_btn.get('x', False),
                               'y': cur_btn['y'] and not prev_btn.get('y', False)}
                prev_btn = dict(cur_btn)
                ly, lx, rx = get_remote_axes(remote)

            if just_pressed.get('y', False):
                if not use_real_robot:
                    reset_robot(robot.data)
                    mujoco.mj_forward(robot.model, robot.data)
                state, sit_stand_t, t_global = 'SIT', 0.0, 0.0
                ekf_warmup_elapsed = 0.0
                ekf.reset(sit_height)
                print('Robot reset')

            if just_pressed.get('a', False):
                state = 'RISING' if state in ('SIT', 'LOWERING') else 'LOWERING'

            if state in ('STAND', 'TROT'):
                state = 'TROT' if (abs(ly) > 0 or abs(lx) > 0 or abs(rx) > 0) else 'STAND'

            if use_real_robot:
                dt = 0.002
                t_global += dt
                if not ekf.enabled:
                    ekf_warmup_elapsed += dt
                    if ekf_warmup_elapsed >= EKF_WARMUP_SECS:
                        ekf.enabled = True
                        print('EKF enabled.')
                z_accel = robot.low_state.imu_state.accelerometer[2]
                ekf.predict(z_accel)
            else:
                dt = robot.model.opt.timestep
                t_global += dt
                if not ekf.enabled:
                    ekf_warmup_elapsed += dt
                    if ekf_warmup_elapsed >= EKF_WARMUP_SECS:
                        ekf.enabled = True
                        print('EKF enabled.')
                z_accel = robot.read_imu_z_accel()
                ekf.predict(z_accel)

            if state == 'RISING':
                sit_stand_t = min(1.0, sit_stand_t + dt / TRANSITION_DURATION)
                if sit_stand_t >= 1.0:
                    state = 'STAND'
            elif state == 'LOWERING':
                sit_stand_t = max(0.0, sit_stand_t - dt / TRANSITION_DURATION)
                if sit_stand_t <= 0.0:
                    state = 'SIT'

            targets = [0.0] * 12
            for leg in range(4):
                if state == 'TROT':
                    ph = (STEP_FREQ * 2.0 * math.pi * t_global + PHASE_OFFSET[leg]) % (2.0 * math.pi)
                    stride_x = FWD_AXIS_SIGN * ly * STEP_LEN_X
                    stride_y = lx * STEP_LEN_Y
                    hip_t, thigh_t, calf_t = compute_foot_target(leg, ph, stride_x, stride_y, rx)
                    if not use_real_robot and ph >= math.pi:
                        ekf.update(robot.data.qpos[2])
                elif state in ('RISING', 'LOWERING'):
                    hip_t, thigh_t, calf_t = lerp_pose(_LEG_KEYS[leg], sit_stand_t)
                elif state == 'SIT':
                    hip_t, thigh_t, calf_t = REAL_SIT[_LEG_KEYS[leg]]
                else:
                    hip_t, thigh_t, calf_t = REAL_STAND[_LEG_KEYS[leg]]

                qp = QPOS_IDX[leg]
                qv = QVEL_IDX[leg]
                idx_base = leg * 3
                targets[idx_base:idx_base + 3] = [hip_t, thigh_t, calf_t]

                if use_real_robot:
                    # The robot controller expects a target angle per joint, with feedforward torque
                    # from the low-level PID loop. This replaced the simulator-only qpos/qvel loop.
                    pass

            if use_real_robot:
                robot.set_target_angles(targets, feedforward_torques=[0.0] * 12)
            else:
                robot.set_joint_targets(targets)
                robot.step()
                if viewer is not None:
                    viewer.sync()

            time.sleep(max(0.0, 0.002 - (time.perf_counter() - wall_origin) % 0.002))

    except KeyboardInterrupt:
        print('Stopping robot controller')
    finally:
        if viewer is not None:
            viewer.close()
        if pygame is not None:
            pygame.quit()


if __name__ == '__main__':
    # To run in simulation: python go2_PID_robot.py
    # To observe the real robot without commands:
    #   python go2_PID_robot.py --real
    # To enable motor commands after validation:
    #   python go2_PID_robot.py --real --enable-motors
    use_real_robot = '--real' in sys.argv
    enable_motors = '--enable-motors' in sys.argv
    main(use_real_robot=use_real_robot, enable_motors=enable_motors)
