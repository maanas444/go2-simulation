"""
Go2 MuJoCo Controller — Xbox 360 Afterglow
===========================================
A = SIT/STAND toggle (smooth interpolation)
X = JUMP (only when standing/trotting)
Y = RESET to sit

Left stick = walk / strafe
Right stick X = turn

Standing and sitting poses calibrated from real hardware measurements.
Starts in sitting position. A button smoothly interpolates to stand and back.

Button detection happens once per render frame (not per 2ms physics step)
so presses are never missed.

Single-threaded. Before each viewer.sync() we step physics enough times
to match wall-clock elapsed time. sync() costs ~18ms → ~9 steps per frame
→ sim runs at exactly 1:1 realtime. No threading, no MuJoCo data races.
"""

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

# ── Real robot standing pose (measured from hardware) ─────────────────────────
REAL_STAND = {
    "FR": ( 0.018,  0.667, -1.377),
    "FL": (-0.018,  0.663, -1.369),
    "RR": ( 0.085,  0.660, -1.353),
    "RL": (-0.082,  0.658, -1.351),
}

# ── Real robot sitting pose (measured from hardware) ──────────────────────────
REAL_SIT = {
    "FR": ( 0.061,  1.236, -2.761),
    "FL": (-0.068,  1.241, -2.770),
    "RR": ( 0.383,  1.243, -2.756),
    "RL": (-0.402,  1.244, -2.758),
}

# ── Symmetrised stand averages for gait IK ───────────────────────────────────
HIP_STAND   =  0.0
THIGH_STAND =  0.662
CALF_STAND  = -1.363

FOOT_Z_STAND = -(L_THIGH * math.cos(THIGH_STAND) +
                 L_CALF  * math.cos(THIGH_STAND + CALF_STAND))
print(f"[init] Effective FOOT_Z_STAND: {FOOT_Z_STAND:.4f}")

# ── Sit/stand transition ──────────────────────────────────────────────────────
TRANSITION_DURATION = 1.5   # seconds for full sit→stand or stand→sit

# ── Gait ─────────────────────────────────────────────────────────────────────
STEP_FREQ   =  2.0
STEP_HEIGHT =  0.08
STEP_LEN_X  =  0.15
TURN_STRIDE =  0.08
PHASE_OFFSET = [0.0, math.pi, math.pi, 0.0]   # FR FL RR RL

# ── Jump ─────────────────────────────────────────────────────────────────────
JUMP_CROUCH_Z = -0.18
JUMP_LAUNCH_Z = -0.38
JUMP_TUCK_Z   = -0.22
JUMP_LAND_Z   = -0.22
JUMP_TIMINGS  = {"CROUCH": 0.20, "LAUNCH": 0.12, "FLIGHT": 0.35, "LAND": 0.30}
JUMP_SEQ      = ["CROUCH", "LAUNCH", "FLIGHT", "LAND"]

# ── PD gains ─────────────────────────────────────────────────────────────────
KP         = [100.0, 250.0, 200.0]
KD         = [  3.0,   6.0,   5.0]
TORQUE_MAX = [ 23.7,  23.7,  45.43]

# ── Index maps ────────────────────────────────────────────────────────────────
CTRL_IDX = [[0,1,2],[3,4,5],[6,7,8],[9,10,11]]
QPOS_IDX = [[10,11,12],[7,8,9],[16,17,18],[13,14,15]]
QVEL_IDX = [[ 9,10,11],[6,7,8],[15,16,17],[12,13,14]]

# ── Controller bindings ───────────────────────────────────────────────────────
AXIS_LX = 0; AXIS_LY = 1; AXIS_RX = 3
BTN_A = 0; BTN_X = 2; BTN_Y = 3
DEADZONE = 0.12

_LEG_KEYS = ["FR", "FL", "RR", "RL"]


# ── IK ────────────────────────────────────────────────────────────────────────
def ik(px, pz):
    r = math.sqrt(px*px + pz*pz)
    r = float(np.clip(r, 0.05, L_THIGH + L_CALF - 0.005))
    cos_c = (L_THIGH**2 + L_CALF**2 - r**2) / (2.0*L_THIGH*L_CALF)
    calf  = -(math.pi - math.acos(float(np.clip(cos_c, -1.0, 1.0))))
    alpha = math.atan2(px, -pz)
    cos_b = (L_THIGH**2 + r**2 - L_CALF**2) / (2.0*L_THIGH*r)
    thigh = alpha + math.acos(float(np.clip(cos_b, -1.0, 1.0)))
    return (float(np.clip(thigh, *THIGH_LIM)),
            float(np.clip(calf,  *CALF_LIM)))


POSE = {
    "CROUCH": (0.0, *ik(0.0, JUMP_CROUCH_Z)),
    "LAUNCH": (0.0, *ik(0.0, JUMP_LAUNCH_Z)),
    "FLIGHT": (0.0, *ik(0.0, JUMP_TUCK_Z)),
    "LAND":   (0.0, *ik(0.0, JUMP_LAND_Z)),
}


def lerp_pose(leg_key, t):
    """Smooth ease-in-out interpolation between sit (t=0) and stand (t=1)."""
    s     = t * t * (3.0 - 2.0 * t)
    sit   = REAL_SIT[leg_key]
    stand = REAL_STAND[leg_key]
    return (
        sit[0] + s * (stand[0] - sit[0]),
        sit[1] + s * (stand[1] - sit[1]),
        sit[2] + s * (stand[2] - sit[2]),
    )


def reset_pose(data, sit_stand_t=0.0):
    """Reset physics to current sit/stand interpolation position."""
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[2] = 0.25 + sit_stand_t * (0.35 - 0.25)
    data.qpos[3] = 1.0
    for leg in range(4):
        hip, thigh, calf = lerp_pose(_LEG_KEYS[leg], sit_stand_t)
        qp = QPOS_IDX[leg]
        data.qpos[qp[0]] = hip
        data.qpos[qp[1]] = thigh
        data.qpos[qp[2]] = calf


def pd(target, pos, vel, j):
    return float(np.clip(KP[j]*(target - pos) - KD[j]*vel,
                         -TORQUE_MAX[j], TORQUE_MAX[j]))


def dz(v):
    return v if abs(v) > DEADZONE else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pygame.init(); pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No controller."); return
    joy = pygame.joystick.Joystick(0); joy.init()
    print(f"Controller: {joy.get_name()}")
    print("A = SIT/STAND toggle")
    print("X = JUMP (standing only)   Y = RESET")
    print("Left stick = move   Right stick X = turn")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    model.opt.timestep   = 0.002
    model.opt.iterations = 50
    model.opt.solver     = mujoco.mjtSolver.mjSOL_CG

    reset_pose(data)
    mujoco.mj_forward(model, data)

    # ── State: SIT | RISING | STAND | TROT | LOWERING | JUMP ─────────────────
    state       = "SIT"
    jump_phase  = 0
    jump_t      = 0.0
    t_gait      = 0.0
    sit_stand_t = 0.0          # 0.0 = sit, 1.0 = stand

    # Previous button states — sampled ONCE per render frame, before physics loop
    prev_btn = {BTN_A: False, BTN_X: False, BTN_Y: False}

    sim_time    = 0.0
    wall_origin = time.perf_counter()

    DT     = model.opt.timestep
    T_STEP = DT / TRANSITION_DURATION

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = 2.5
        viewer.cam.elevation = -15

        while viewer.is_running():

            # ── Sample inputs ONCE per render frame ───────────────────────
            # This is the fix: pump events here, outside the physics loop,
            # so button edges are never missed across 500Hz steps.
            pygame.event.pump()
            cur_btn = {b: bool(joy.get_button(b)) for b in (BTN_A, BTN_X, BTN_Y)}
            just_pressed = {b: cur_btn[b] and not prev_btn[b] for b in cur_btn}
            prev_btn = dict(cur_btn)

            ly = dz(-joy.get_axis(AXIS_LY))
            lx = dz( joy.get_axis(AXIS_LX))
            rx = dz(-joy.get_axis(AXIS_RX))

            # ── Handle button events ──────────────────────────────────────
            if just_pressed[BTN_A] and state != "JUMP":
                if state in ("SIT", "LOWERING"):
                    state = "RISING"
                    print("→ RISING")
                elif state in ("STAND", "RISING", "TROT"):
                    t_gait = 0.0
                    state  = "LOWERING"
                    print("→ LOWERING")

            if just_pressed[BTN_X] and state in ("STAND", "TROT"):
                state = "JUMP"; jump_phase = 0; jump_t = sim_time
                print("→ JUMP")

            if just_pressed[BTN_Y]:
                reset_pose(data, sit_stand_t); mujoco.mj_forward(model, data)
                sim_time = 0.0; wall_origin = time.perf_counter()
                t_gait = 0.0
                print("→ RESET (state kept)")

            # ── Auto trot/stand from sticks ───────────────────────────────
            if state == "STAND":
                if abs(ly) > 0 or abs(lx) > 0 or abs(rx) > 0:
                    state = "TROT"
            elif state == "TROT":
                if abs(ly) == 0 and abs(lx) == 0 and abs(rx) == 0:
                    state = "STAND"

            # ── Step physics to catch up to wall clock ────────────────────
            wall_now     = time.perf_counter()
            wall_elapsed = wall_now - wall_origin
            target_sim   = min(wall_elapsed, sim_time + 0.050)

            while sim_time < target_sim:

                # ── Advance sit/stand interpolation ──────────────────────
                if state == "RISING":
                    sit_stand_t = min(1.0, sit_stand_t + T_STEP)
                    if sit_stand_t >= 1.0:
                        state = "STAND"
                        print("→ STAND")
                elif state == "LOWERING":
                    sit_stand_t = max(0.0, sit_stand_t - T_STEP)
                    if sit_stand_t <= 0.0:
                        state = "SIT"
                        print("→ SIT")

                # ── Jump phase sequencer ──────────────────────────────────
                if state == "JUMP":
                    phase_name = JUMP_SEQ[jump_phase]
                    if sim_time - jump_t >= JUMP_TIMINGS[phase_name]:
                        jump_phase += 1
                        if jump_phase >= len(JUMP_SEQ):
                            state = "STAND"; jump_phase = 0
                            print("→ STAND (landed)")
                        else:
                            jump_t = sim_time
                            print(f"  jump: {JUMP_SEQ[jump_phase]}")

                if state == "TROT":
                    t_gait += DT

                # ── Compute joint targets ─────────────────────────────────
                w = STEP_FREQ * 2.0 * math.pi

                for leg in range(4):
                    side = 1.0 if leg in (0, 2) else -1.0

                    if state == "TROT":
                        ph = (w * t_gait + PHASE_OFFSET[leg]) % (2.0 * math.pi)
                        s, c = math.sin(ph), math.cos(ph)
                        sx = ly * STEP_LEN_X + side * rx * TURN_STRIDE
                        if s >= 0:
                            prog = (1.0 - c) / 2.0
                            px = sx * (1.0 - 2.0 * prog)
                            pz = FOOT_Z_STAND + STEP_HEIGHT * math.sin(math.pi * prog)
                        else:
                            prog = (1.0 + c) / 2.0
                            px = -sx * (1.0 - 2.0 * prog)
                            pz = FOOT_Z_STAND
                        hip_t           = float(np.clip(lx * 0.10 * side, *HIP_LIM))
                        thigh_t, calf_t = ik(px, pz)

                    elif state == "JUMP":
                        hip_t, thigh_t, calf_t = POSE[JUMP_SEQ[jump_phase]]

                    elif state in ("RISING", "LOWERING"):
                        hip_t, thigh_t, calf_t = lerp_pose(_LEG_KEYS[leg], sit_stand_t)

                    elif state == "STAND":
                        hip_t, thigh_t, calf_t = REAL_STAND[_LEG_KEYS[leg]]

                    else:  # SIT
                        hip_t, thigh_t, calf_t = REAL_SIT[_LEG_KEYS[leg]]

                    qp = QPOS_IDX[leg]; qv = QVEL_IDX[leg]; ci = CTRL_IDX[leg]
                    data.ctrl[ci[0]] = pd(hip_t,   data.qpos[qp[0]], data.qvel[qv[0]], 0)
                    data.ctrl[ci[1]] = pd(thigh_t, data.qpos[qp[1]], data.qvel[qv[1]], 1)
                    data.ctrl[ci[2]] = pd(calf_t,  data.qpos[qp[2]], data.qvel[qv[2]], 2)

                mujoco.mj_step(model, data)
                sim_time += DT

            # ── Render ────────────────────────────────────────────────────
            viewer.sync()


if __name__ == "__main__":
    main()