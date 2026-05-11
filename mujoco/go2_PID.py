import mujoco
import mujoco.viewer
import pygame
import time, os, math
import numpy as np

XML_PATH = os.path.expanduser('~/unitree_mujoco/unitree_robots/go2/scene.xml')

# ── Robot geometry ────────────────────────────────────────────────────────────
L_THIGH = 0.213
L_CALF  = 0.213
THIGH_LIM = (-1.571,  3.491)
CALF_LIM  = (-2.723, -0.838)
HIP_LIM   = (-1.047,  1.047)

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

# ── Controller/Gait Config ───────────────────────────────────────────────────
TRANSITION_DURATION = 1.5
STEP_FREQ   = 2.0
STEP_HEIGHT = 0.08
STEP_LEN_X  = 0.18 # Slightly increased for better stride
TURN_STRIDE = 0.12 # Increased for noticeable pivoting
PHASE_OFFSET = [0.0, math.pi, math.pi, 0.0] # FR FL RR RL

# ── PID gains ─────────────────────────────────────────────────────────────────
KP, KI, KD = [100.0, 250.0, 200.0], [5.0, 10.0, 10.0], [3.0, 6.0, 5.0]
TORQUE_MAX = [23.7, 23.7, 45.43]

# ── Index maps ────────────────────────────────────────────────────────────────
CTRL_IDX = [[0,1,2],[3,4,5],[6,7,8],[9,10,11]]
QPOS_IDX = [[10,11,12],[7,8,9],[16,17,18],[13,14,15]]
QVEL_IDX = [[ 9,10,11],[6,7,8],[15,16,17],[12,13,14]]

AXIS_LX, AXIS_LY, AXIS_RX = 0, 1, 3
BTN_A, BTN_X, BTN_Y, DEADZONE = 0, 2, 3, 0.12
_LEG_KEYS = ["FR", "FL", "RR", "RL"]

class StateEstimator:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.array([[0.28], [0.0]])
        self.P = np.eye(2) * 0.5
        self.Q = np.diag([0.001, 0.001]) 
        self.R = 0.01                    

    def predict(self, z_accel):
        a_world = z_accel - 9.81
        F = np.array([[1, self.dt], [0, 1]])
        B = np.array([[0.5 * self.dt**2], [self.dt]])
        self.x = F @ self.x + B * a_world
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measured_z):
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
    r = np.clip(math.sqrt(px*px + pz*pz), 0.05, L_THIGH + L_CALF - 0.005)
    cos_c = (L_THIGH**2 + L_CALF**2 - r**2) / (2.0*L_THIGH*L_CALF)
    calf = -(math.pi - math.acos(np.clip(cos_c, -1.0, 1.0)))
    alpha = math.atan2(px, -pz)
    cos_b = (L_THIGH**2 + r**2 - L_CALF**2) / (2.0*L_THIGH*r)
    thigh = alpha + math.acos(np.clip(cos_b, -1.0, 1.0))
    return float(np.clip(thigh, *THIGH_LIM)), float(np.clip(calf, *CALF_LIM))

def lerp_pose(leg_key, t):
    s = t * t * (3.0 - 2.0 * t)
    sit, stand = REAL_SIT[leg_key], REAL_STAND[leg_key]
    return tuple(sit[i] + s * (stand[i] - sit[i]) for i in range(3))

def dz(v): return v if abs(v) > DEADZONE else 0.0

def reset_robot(data):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[2] = 0.12 # Start low for SIT
    data.qpos[3] = 1.0
    # Apply initial joint positions
    for leg in range(4):
        hip, thigh, calf = REAL_SIT[_LEG_KEYS[leg]]
        qp = QPOS_IDX[leg]
        data.qpos[qp[0]], data.qpos[qp[1]], data.qpos[qp[2]] = hip, thigh, calf

def main():
    pygame.init(); pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No Joystick found!"); return
    joy = pygame.joystick.Joystick(0); joy.init()

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002
    
    pids = [[PIDController(KP[j], KI[j], KD[j], TORQUE_MAX[j]) for j in range(3)] for _ in range(4)]
    ekf = StateEstimator(model.opt.timestep)

    reset_robot(data)
    mujoco.mj_forward(model, data)

    state, t_global, sit_stand_t = "SIT", 0.0, 0.0
    prev_btn = {BTN_A: False, BTN_X: False, BTN_Y: False}
    sim_time, wall_origin = 0.0, time.perf_counter()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            pygame.event.pump()
            cur_btn = {b: bool(joy.get_button(b)) for b in (BTN_A, BTN_X, BTN_Y)}
            just_pressed = {b: cur_btn[b] and not prev_btn[b] for b in cur_btn}
            prev_btn = dict(cur_btn)

            # --- FIX 1: Corrected Stick Polarity ---
            ly = dz(joy.get_axis(AXIS_LY))   # Fwd/Bwd (inverted signs removed)
            lx = dz(joy.get_axis(AXIS_LX))   # Strafe
            rx = dz(-joy.get_axis(AXIS_RX))  # Turn

            # --- FIX 2: Reset Button Logic ---
            if just_pressed[BTN_Y]:
                reset_robot(data)
                mujoco.mj_forward(model, data)
                state, sit_stand_t, t_global = "SIT", 0.0, 0.0
                print("Robot Reset")

            if just_pressed[BTN_A]:
                state = "RISING" if state in ("SIT", "LOWERING") else "LOWERING"
            
            # --- FIX 3: Persistence of State for Pivoting ---
            if state in ("STAND", "TROT"):
                if abs(ly) > 0 or abs(lx) > 0 or abs(rx) > 0:
                    state = "TROT"
                else:
                    state = "STAND"

            target_sim = min(time.perf_counter() - wall_origin, sim_time + 0.050)
            while sim_time < target_sim:
                DT = model.opt.timestep
                t_global += DT 
                
                z_accel = data.sensor("imu_acc").data[2]
                ekf.predict(z_accel)

                if state == "RISING":
                    sit_stand_t = min(1.0, sit_stand_t + DT/TRANSITION_DURATION)
                    if sit_stand_t >= 1.0: state = "STAND"
                elif state == "LOWERING":
                    sit_stand_t = max(0.0, sit_stand_t - DT/TRANSITION_DURATION)
                    if sit_stand_t <= 0.0: state = "SIT"

                for leg in range(4):
                    side = 1.0 if leg in (0, 2) else -1.0 # Left vs Right
                    
                    if state == "TROT":
                        ph = (STEP_FREQ * 2.0 * math.pi * t_global + PHASE_OFFSET[leg]) % (2.0 * math.pi)
                        
                        # --- FIX 4: Mapping pivoting to Trajectory ---
                        stride_x = ly * STEP_LEN_X
                        stride_y = lx * 0.08
                        # Yaw adds to X stride differently for front/back legs to pivot
                        yaw_factor = 1.0 if leg in (0, 1) else -1.0
                        yaw_offset = rx * TURN_STRIDE * yaw_factor

                        if ph < math.pi: # SWING
                            prog = ph / math.pi
                            px = -stride_x/2 + stride_x * prog + yaw_offset * math.sin(math.pi * prog)
                            py = -stride_y/2 + stride_y * prog
                            pz = FOOT_Z_STAND + STEP_HEIGHT * math.sin(math.pi * prog)
                        else: # STANCE
                            prog = (ph - math.pi) / math.pi
                            px = stride_x/2 - stride_x * prog - yaw_offset * math.sin(math.pi * prog)
                            py = stride_y/2 - stride_y * prog
                            pz = FOOT_Z_STAND
                            ekf.update(0.28)

                        hip_t, (thigh_t, calf_t) = py, ik(px, pz)
                        
                    elif state in ("RISING", "LOWERING"):
                        hip_t, thigh_t, calf_t = lerp_pose(_LEG_KEYS[leg], sit_stand_t)
                    elif state == "SIT":
                        hip_t, thigh_t, calf_t = REAL_SIT[_LEG_KEYS[leg]]
                    else: # STAND
                        hip_t, thigh_t, calf_t = REAL_STAND[_LEG_KEYS[leg]]

                    qp, qv, ci = QPOS_IDX[leg], QVEL_IDX[leg], CTRL_IDX[leg]
                    targets = [hip_t, thigh_t, calf_t]
                    for j in range(3):
                        data.ctrl[ci[j]] = pids[leg][j].update(targets[j], data.qpos[qp[j]], data.qvel[qv[j]], DT)

                mujoco.mj_step(model, data)
                sim_time += DT
            viewer.sync()

if __name__ == "__main__":
    main()