import time
import pygame
import mujoco
import mujoco.viewer
import numpy as np

# Import your custom modules
from output import MujocoOutput, UnitreeGo2Output
from gait_engine import GaitEngine 
from state_estimator import StateEstimator 

# --- CONFIG & CONSTANTS ---
USE_REAL_ROBOT = True  # Toggle this for sim vs reality
DT = 0.002              # 500Hz loop
XML_PATH = "scene.xml"  # Path to your MuJoCo model

# These should match your previous go2_PID logic
CTRL_IDX = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
QPOS_IDX = [[7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]
QVEL_IDX = [[6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17]]

def main():
    # 1. INITIALIZE HARDWARE/SIM INTERFACE
    if USE_REAL_ROBOT:
        robot = UnitreeGo2Output(network_interface="eth0")
        model, data = None, None # Not needed for real robot
    else:
        # Load MuJoCo
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
        # Placeholder for your PID objects (assuming a simple list here)
        # In a real setup, you'd initialize your PID class for each joint
        pids = [[None]*3 for _ in range(4)] 
        robot = MujocoOutput(model, data, pids, CTRL_IDX, QPOS_IDX, QVEL_IDX)
    
    robot.connect()

    # 2. INITIALIZE LOGIC & INPUT
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        joy = pygame.joystick.Joystick(0)
        joy.init()
    else:
        print("Warning: No joystick detected!")
        joy = None

    ekf = StateEstimator(DT)
    gait = GaitEngine() 
    
    print("System Online. Press Ctrl+C to stop.")

    # 3. THE MAIN LOOP
    # If in sim, use the viewer; if real, use a standard while True
    try:
        while True:
            start_time = time.perf_counter()

            # --- STEP 1: READ INPUTS ---
            pygame.event.pump()
            joy_lx = joy.get_axis(1) if joy else 0.0
            joy_ly = joy.get_axis(0) if joy else 0.0
            joy_rx = joy.get_axis(3) if joy else 0.0

            # --- STEP 2: SENSOR DATA (Defining z_accel) ---
            if USE_REAL_ROBOT:
                # Pull from Unitree SDK LowState (Requires your robot_io to expose this)
                z_accel = robot.low_state.imu_state.accelerometer[2]
            else:
                # Pull from MuJoCo IMU sensor (Index 2 is Z)
                # Ensure your XML has an <accelerometer> sensor defined!
                z_accel = data.sensor("imu_acc").data[2]

            # --- STEP 3: STATE ESTIMATION ---
            # Prediction using IMU
            ekf.predict(z_accel)
            
            # Correction using "Measured Z"
            # We assume height is 0.28m when the robot is standing/trotting
            # Ideally, gait.is_stance() tells you if the feet are on the floor
            if abs(joy_lx) < 0.1 and abs(joy_ly) < 0.1: # If standing still
                measured_z = 0.28 
                ekf.update(measured_z)

            # --- STEP 4: GAIT CALCULATION (Friend's Logic) ---
            # Passing your filtered EKF height into their math
            target_angles = gait.update(joy_lx, joy_ly, joy_rx, ekf.height)

            # --- STEP 5: OUTPUT ---
            # Send 12 angles. Feedforward torque is 0.0 for now.
            robot.send_commands(target_angles, [0.0]*12)

            # --- STEP 6: TIMING & SIM STEP ---
            if not USE_REAL_ROBOT:
                mujoco.mj_step(model, data)

            elapsed = time.perf_counter() - start_time
            if elapsed < DT:
                time.sleep(DT - elapsed)

    except KeyboardInterrupt:
        print("\nShutting down safely...")

if __name__ == "__main__":
    main()