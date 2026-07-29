import mujoco
import mujoco.viewer
import pygame
import time, os, math
import numpy as np
import numpy.typing as npt
from typing import Tuple

# ── Mocked Data Structures (to replace missing imports) ───────────────────────
class AngleLimits:
    def __init__(self, min_val, max_val):
        self.minimum = min_val
        self.maximum = max_val

Point2D = npt.NDArray[np.float64]
Point3D = npt.NDArray[np.float64]
Vector = npt.NDArray[np.float64]
Point3DList = list
LegPoseList = npt.NDArray[np.float64]
LEG_COUNT = 4
type LegPose = Tuple[float, float, float]


# ── IK / FK Configuration ─────────────────────────────────────────────────────
# Zero offsets for angles #
_ANGLE_ZERO_OFFSETS = np.array([
     np.pi/2,  # Abductor 90 deg
    -np.pi/2,  # Hip     -90 deg
     0,        # Knee      0 deg
])  # In Radians

# Leg offsets from body origin #
_LEG_OFFSETS_FROM_BODY_ORIGIN = np.array([
    [ 0.1934,  0.0465, 0.0],  # FL
    [ 0.1934, -0.0465, 0.0],  # FR
    [-0.1934,  0.0465, 0.0],  # BL
    [-0.1934, -0.0465, 0.0]   # BR
], dtype=float)

# Link Lengths in meters #
_HIP_OFFSET   = 0.01  # CANNOT BE ZERO
_THIGH_LENGTH = 0.213
_CALF_LENGTH  = 0.213

# Rotation limits (min, max) in Radians #
_HIP_ABDUCTOR_ROT_RANGE = AngleLimits(-1.0472,  1.0472)   
_FRONT_HIP_ROT_RANGE    = AngleLimits(-1.5708,  3.4907)   
_BACK_HIP_ROT_RANGE     = AngleLimits(-0.5236,  4.5379)   
_KNEE_ROT_RANGE         = AngleLimits(-2.7227, -0.83776)  

# Output torque limits in Newton-meters #
_KNEE_TORQUE_LIMIT = (-45.43, 45.43)

# Accuracy #
_INPUT_ACCURACY = 5  
_ANGLE_ACCURACY = 5  

# Reachability limits #
_MAX_RANGE_LENGTH = np.sqrt(np.square(_THIGH_LENGTH) + np.square(_CALF_LENGTH) - (2 * _THIGH_LENGTH * _CALF_LENGTH * np.cos(np.pi - _KNEE_ROT_RANGE.maximum)))
_MAX_RANGE_LENGTH = np.round(_MAX_RANGE_LENGTH, _INPUT_ACCURACY)


# ── Shared Math Helpers ───────────────────────────────────────────────────────
def get_unit_vectors_of_a_plane(normal_vector:Vector) -> Tuple[Vector, Vector]:
    n_raw = normal_vector
    magnitude_n = np.linalg.norm(n_raw)
    
    if np.isclose(magnitude_n, 0):  
        raise ValueError("Normal vector cannot be a zero vector.")

    n_unit = n_raw / magnitude_n
    a, b, _ = n_unit

    if np.isclose(a, 0) and np.isclose(b, 0):  
        u_raw = np.array([1, 0, 0])
    else:
        u_raw = np.array([-b, a, 0])

    magnitude_u = np.linalg.norm(u_raw)
    u_unit = u_raw / magnitude_u

    v_unit = np.cross(n_unit, u_unit) 
    return u_unit, v_unit


# ── Inverse Kinematics ────────────────────────────────────────────────────────
class IK_Solver:
    def __init__(self):
        pass

    def _solve_internal(self, leg_origin:Point3D, point:Point3D) -> None|LegPose:
        delta_point = point - leg_origin

        _, delta_y, delta_z = delta_point
        c = np.hypot(delta_y, delta_z)
        if c < _HIP_OFFSET:
            print(f"IK failed - unreachable point! (Too close : {c} < {_HIP_OFFSET})")
            return None  
        
        alpha = np.arctan2(delta_z, delta_y) + _ANGLE_ZERO_OFFSETS[0]
        beta  = np.arccos((_HIP_OFFSET / c))
        
        abductor_angle = (alpha - beta) + _ANGLE_ZERO_OFFSETS[0]

        movement_normal:Vector = np.array([
            0,
            _HIP_OFFSET * np.cos(abductor_angle),
            _HIP_OFFSET * np.sin(abductor_angle)
        ])

        plane_origin:Point3D = movement_normal
        u_unit, v_unit = get_unit_vectors_of_a_plane(movement_normal)

        point_relative_to_plane = delta_point - plane_origin
        local_x = np.dot(point_relative_to_plane, u_unit)
        local_z = np.dot(point_relative_to_plane, v_unit)

        r = np.hypot(local_x, local_z) 
        if np.round(r, _INPUT_ACCURACY) > _MAX_RANGE_LENGTH:
            print(f"IK failed - unreachable point! (Too far : {np.round(r, _INPUT_ACCURACY)} > {_MAX_RANGE_LENGTH})")
            return None 

        phi   = np.arctan2(local_z, local_x)
        psi   = np.arccos((np.square(_THIGH_LENGTH) + np.square(_CALF_LENGTH) - np.square(r)) / (2 * _THIGH_LENGTH * _CALF_LENGTH))  
        gamma = np.arccos((np.square(_THIGH_LENGTH) + np.square(r) - np.square(_CALF_LENGTH)) / (2 * _THIGH_LENGTH * r))             

        hip_angle   = (gamma + phi) - _ANGLE_ZERO_OFFSETS[1]
        knee_angle  = (psi - np.pi) 

        return abductor_angle, hip_angle, knee_angle

    def _solve_leg(self, leg_origin:Point3D, point:Point3D, is_front_leg:bool) -> None|LegPose:
        result = self._solve_internal(leg_origin, point)
        if result is None:
            return None
        
        abductor_angle, hip_angle, knee_angle = result

        rounded_abductor_angle = np.round(abductor_angle, _ANGLE_ACCURACY)
        rounded_hip_angle      = np.round(hip_angle, _ANGLE_ACCURACY)
        rounded_knee_angle     = np.round(knee_angle, _ANGLE_ACCURACY)
        
        if rounded_abductor_angle < _HIP_ABDUCTOR_ROT_RANGE.minimum:
            print(f"IK fail - joint angle limit hit! (abductor.min; {rounded_abductor_angle} >= {_HIP_ABDUCTOR_ROT_RANGE.minimum})")
            return None
        if rounded_abductor_angle > _HIP_ABDUCTOR_ROT_RANGE.maximum:
            print(f"IK fail - joint angle limit hit! (abductor.max; {rounded_abductor_angle} <= {_HIP_ABDUCTOR_ROT_RANGE.maximum})")
            return None
        
        _hip_rot_range = _FRONT_HIP_ROT_RANGE if is_front_leg else _BACK_HIP_ROT_RANGE
        if rounded_hip_angle < _hip_rot_range.minimum:
            print(f"IK fail - joint angle limit hit! (hip.min; {rounded_hip_angle} >= {_hip_rot_range.minimum})")
            return None
        if rounded_hip_angle > _hip_rot_range.maximum:
            print(f"IK fail - joint angle limit hit! (hip.max; {rounded_hip_angle} <= {_hip_rot_range.maximum})")
            return None
        
        if rounded_knee_angle < _KNEE_ROT_RANGE.minimum:
            print(f"IK fail - joint angle limit hit! (knee.min; {rounded_knee_angle} >= {_KNEE_ROT_RANGE.minimum})")
            return None
        if rounded_knee_angle > _KNEE_ROT_RANGE.maximum:
            print(f"IK fail - joint angle limit hit! (knee.max; {rounded_knee_angle} <= {_KNEE_ROT_RANGE.maximum})")
            return None

        return result

    def solve(self, leg_points:Point3DList) -> LegPoseList:
        if len(leg_points) > LEG_COUNT:
            raise IndexError(f"Too many leg points! ({len(leg_points)} == {LEG_COUNT})")
        if len(leg_points) < LEG_COUNT:
            raise IndexError(f"Not enough leg points! ({len(leg_points)} == {LEG_COUNT})")

        common_origin:Point3D = np.array([0, 0, 0], dtype=np.float64)

        pose_accumulator = []
        for i in range(LEG_COUNT):
            pose_accumulator.append(self._solve_leg(common_origin, leg_points[i], (i < (LEG_COUNT // 2))))  

        leg_poses: LegPoseList = np.array(pose_accumulator, dtype=np.float64)
        return leg_poses


# ── Forward Kinematics ────────────────────────────────────────────────────────
def degrees_to_radians(deg:float) -> float:
    return deg * (np.pi / 180)

def _polar_to_cartesian_coordinate(distance:float, angle:float) -> Point2D:
    return np.array([
        distance * np.cos(angle),
        distance * np.sin(angle)
    ])

def _spherical_to_cartesian_coordinate(distance_r:float, azimuth_angle:float, polar_angle:float, start_point:Point3D|None = None) -> Point3D:
    if start_point is None:
        start_point = np.array([0,0,0])

    return start_point + np.array([
        (distance_r * np.sin(polar_angle) * np.cos(azimuth_angle)),
        (distance_r * np.sin(polar_angle) * np.sin(azimuth_angle)),
        (distance_r * np.cos(polar_angle)),
    ])

def _convert_local_to_world_coordinate(local_coordinate:Point2D, plane_anchor_point:Point3D, u_unit:Vector, v_unit:Vector) -> Point3D:
    x_prime, y_prime = local_coordinate
    return plane_anchor_point + (x_prime * u_unit) + (y_prime * v_unit)

def calculate_joint_positions(origin:Point3D, angles:npt.NDArray[np.float64], is_left_side:bool = False):
    AZIMUTH_POS_Y_ANGLE = np.pi/2  
    
    if len(angles) != 3: 
        raise IndexError(f"3 angles must be provided! ({len(angles)} != 3)")
    angles += _ANGLE_ZERO_OFFSETS  
    abductor_angle, hip_angle, knee_relative_angle = angles

    if is_left_side:
        AZIMUTH_POS_Y_ANGLE += np.pi  
        hip_angle           = np.pi - hip_angle
        knee_relative_angle = -knee_relative_angle

    abductor_pos:Point3D = origin

    movement_plane_anchor_point:Point3D = _spherical_to_cartesian_coordinate(_HIP_OFFSET, AZIMUTH_POS_Y_ANGLE, abductor_angle, abductor_pos)
    movement_plane_normal_vector:Vector = movement_plane_anchor_point
    u_unit, v_unit                      = get_unit_vectors_of_a_plane(movement_plane_normal_vector)

    hip_pos:Point3D = movement_plane_anchor_point

    knee_local_pos:Point2D = _polar_to_cartesian_coordinate(_THIGH_LENGTH, hip_angle)
    knee_pos:Point3D       = _convert_local_to_world_coordinate(knee_local_pos, hip_pos, u_unit, v_unit)

    knee_absolute_angle = hip_angle + knee_relative_angle
    foot_local_pos:Point2D = _polar_to_cartesian_coordinate(_CALF_LENGTH, knee_absolute_angle)
    foot_pos:Point3D       = _convert_local_to_world_coordinate(foot_local_pos, knee_pos, u_unit, v_unit)

    return abductor_pos, hip_pos, knee_pos, foot_pos


# ── Robot geometry (Mujoco Simulator Settings) ────────────────────────────────
XML_PATH = os.path.expanduser('~/unitree_mujoco/unitree_robots/go2/scene.xml')

L_THIGH = 0.213
L_CALF  = 0.213

REAL_STAND = {
    "FR": ( 0.018,  0.667, -1.377),
    "FL": (-0.018,  0.663, -1.369),
    "RR": ( 0.085,  0.660, -1.353),
    "RL": (-0.082,  0.658, -1.351),
}

REAL_SIT = {
    "FR": ( 0.061,  1.236, -2.761),
    "FL": (-0.068,  1.241, -2.770),
    "RR": ( 0.383,  1.243, -2.756),
    "RL": (-0.402,  1.244, -2.758),
}

HIP_STAND, THIGH_STAND, CALF_STAND = 0.0, 0.662, -1.363
FOOT_Z_STAND = -(L_THIGH * math.cos(THIGH_STAND) + L_CALF * math.cos(THIGH_STAND + CALF_STAND))

TRANSITION_DURATION = 1.5
STEP_FREQ   = 2.0
STEP_HEIGHT = 0.08
STEP_LEN_X  = 0.18
STEP_LEN_Y  = 0.08
TURN_STRIDE = 0.10
PHASE_OFFSET = [0.0, math.pi, math.pi, 0.0]  # FR FL RR RL
FWD_AXIS_SIGN = -1.0

KP, KI, KD = [100.0, 250.0, 200.0], [5.0, 10.0, 10.0], [3.0, 6.0, 5.0]
TORQUE_MAX = [23.7, 23.7, 45.43]

CTRL_IDX = [[0,1,2],[3,4,5],[6,7,8],[9,10,11]]
QPOS_IDX = [[10,11,12],[7,8,9],[16,17,18],[13,14,15]]
QVEL_IDX = [[ 9,10,11],[6,7,8],[15,16,17],[12,13,14]]

AXIS_LX, AXIS_LY, AXIS_RX = 0, 1, 3
BTN_A, BTN_X, BTN_Y, DEADZONE = 0, 2, 3, 0.12
_LEG_KEYS = ["FR", "FL", "RR", "RL"]

EKF_WARMUP_SECS = 1.5

# Instantiate the refactored global IK Solver
ik_solver = IK_Solver()

# ── Controller & EKF Classes ──────────────────────────────────────────────────
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


def lerp_pose(leg_key, t):
    s = t * t * (3.0 - 2.0 * t)
    sit, stand = REAL_SIT[leg_key], REAL_STAND[leg_key]
    return tuple(sit[i] + s * (stand[i] - sit[i]) for i in range(3))

def dz(v): return v if abs(v) > DEADZONE else 0.0

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
    yaw_stride = yaw * TURN_STRIDE * side_sign
    total_stride_x = stride_x + yaw_stride

    if ph < math.pi:
        prog    = ph / math.pi             
        swing_t = math.sin(math.pi * prog) 

        px = -total_stride_x / 2.0 + total_stride_x * prog
        pz =  FOOT_Z_STAND + STEP_HEIGHT * swing_t
        py_delta = stride_y / 2.0 - stride_y * prog

    else:
        prog = (ph - math.pi) / math.pi    

        px =  total_stride_x / 2.0 - total_stride_x * prog
        pz =  FOOT_Z_STAND
        py_delta = -stride_y / 2.0 + stride_y * prog

    # Factor in the nominal hip offset to keep the target properly relative to the leg base
    py = py_delta + (side_sign * _HIP_OFFSET)
    target_pt = np.array([px, py, pz])
    
    is_front_leg = (leg == 0 or leg == 1)
    
    # Process through the new 3D IK Engine
    result = ik_solver._solve_leg(np.array([0.0, 0.0, 0.0]), target_pt, is_front_leg)
    
    if result is None:
        return REAL_STAND[key]
        
    abductor_angle, hip_angle, knee_angle = result
    return abductor_angle, hip_angle, knee_angle


def main():
    pygame.init(); pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No Joystick found!"); return
    joy = pygame.joystick.Joystick(0); joy.init()

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    model.opt.timestep = 0.002

    pids = [[PIDController(KP[j], KI[j], KD[j], TORQUE_MAX[j]) for j in range(3)] for _ in range(4)]

    sit_height = 0.12   
    ekf = StateEstimator(model.opt.timestep, initial_height=sit_height)

    try:
        ekf_h_adr  = model.sensor('EKF_Height_Est').adr[0]
        true_h_adr = model.sensor('True_Height').adr[0]
        print("Live graphing initialized.")
    except Exception as e:
        print(f"Graphing sensors not found in XML: {e}")
        ekf_h_adr = true_h_adr = None

    reset_robot(data)
    mujoco.mj_forward(model, data)

    state, t_global, sit_stand_t = "SIT", 0.0, 0.0
    ekf_warmup_elapsed = 0.0          
    prev_btn = {BTN_A: False, BTN_X: False, BTN_Y: False}
    sim_time, wall_origin = 0.0, time.perf_counter()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            pygame.event.pump()
            cur_btn  = {b: bool(joy.get_button(b)) for b in (BTN_A, BTN_X, BTN_Y)}
            just_pressed = {b: cur_btn[b] and not prev_btn[b] for b in cur_btn}
            prev_btn = dict(cur_btn)

            ly = dz(joy.get_axis(AXIS_LY))   
            lx = dz(joy.get_axis(AXIS_LX))   
            rx = dz(joy.get_axis(AXIS_RX))   

            if just_pressed[BTN_Y]:
                reset_robot(data)
                mujoco.mj_forward(model, data)
                state, sit_stand_t, t_global = "SIT", 0.0, 0.0
                ekf_warmup_elapsed = 0.0
                ekf.reset(sit_height)
                print("Robot Reset")

            if just_pressed[BTN_A]:
                state = "RISING" if state in ("SIT", "LOWERING") else "LOWERING"

            if state in ("STAND", "TROT"):
                state = "TROT" if (abs(ly) > 0 or abs(lx) > 0 or abs(rx) > 0) else "STAND"

            target_sim = min(time.perf_counter() - wall_origin, sim_time + 0.050)
            while sim_time < target_sim:
                DT = model.opt.timestep
                t_global += DT

                if not ekf.enabled:
                    ekf_warmup_elapsed += DT
                    if ekf_warmup_elapsed >= EKF_WARMUP_SECS:
                        ekf.enabled = True
                        print("EKF enabled.")

                z_accel = data.sensor("imu_acc").data[2]
                ekf.predict(z_accel)

                if ekf_h_adr is not None:
                    data.sensordata[ekf_h_adr] = ekf.x[0][0]
                    data.sensordata[true_h_adr] = data.qpos[2]

                if state == "RISING":
                    sit_stand_t = min(1.0, sit_stand_t + DT / TRANSITION_DURATION)
                    if sit_stand_t >= 1.0: state = "STAND"
                elif state == "LOWERING":
                    sit_stand_t = max(0.0, sit_stand_t - DT / TRANSITION_DURATION)
                    if sit_stand_t <= 0.0: state = "SIT"

                for leg in range(4):
                    if state == "TROT":
                        ph = (STEP_FREQ * 2.0 * math.pi * t_global + PHASE_OFFSET[leg]) % (2.0 * math.pi)

                        stride_x = FWD_AXIS_SIGN * ly * STEP_LEN_X
                        stride_y =  lx * STEP_LEN_Y

                        hip_t, thigh_t, calf_t = compute_foot_target(
                            leg, ph, stride_x, stride_y, rx
                        )

                        if ph >= math.pi:
                            ekf.update(data.qpos[2])  

                    elif state in ("RISING", "LOWERING"):
                        hip_t, thigh_t, calf_t = lerp_pose(_LEG_KEYS[leg], sit_stand_t)
                    elif state == "SIT":
                        hip_t, thigh_t, calf_t = REAL_SIT[_LEG_KEYS[leg]]
                    else:  
                        hip_t, thigh_t, calf_t = REAL_STAND[_LEG_KEYS[leg]]

                    qp, qv, ci = QPOS_IDX[leg], QVEL_IDX[leg], CTRL_IDX[leg]
                    targets = [hip_t, thigh_t, calf_t]
                    for j in range(3):
                        data.ctrl[ci[j]] = pids[leg][j].update(
                            targets[j], data.qpos[qp[j]], data.qvel[qv[j]], DT
                        )

                mujoco.mj_step(model, data)
                sim_time += DT

            viewer.sync()

if __name__ == "__main__":
    main()