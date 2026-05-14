import time
import pygame
from robot_io import MujocoOutput, UnitreeGo2Output
from gait_engine import GaitEngine 
from my_filters import StateEstimator 

# --- CONFIG ---
USE_REAL_ROBOT = False  # The master switch
DT = 0.002             # 500Hz

def main():
    # 1. INITIALIZE OUTPUT
    if USE_REAL_ROBOT:
        robot = UnitreeGo2Output(network_interface="enp3s0")
    else:
        # Set up MuJoCo model/data here...
        robot = MujocoOutput(model, data, pids, CTRL_IDX)
    
    robot.connect()

    # 2. INITIALIZE LOGIC
    ekf = StateEstimator(DT)
    gait = GaitEngine() 
    
    # 3. THE MAIN LOOP
    print("System Online.")
    while True:
        start_time = time.perf_counter()

        # STEP 1: READ INPUTS (Sensors & Joystick)
        # (Get raw IMU and Joint data from the 'robot' object)
        # (Get Joystick axes from Pygame)
        
        # STEP 2: STATE ESTIMATION 
        # ekf.predict(z_accel)
        # if in_stance: ekf.update(-pz)

        # STEP 3: GAIT CALCULATION 
        # target_angles = gait.update(joy_lx, joy_ly, joy_rx, ekf.height)

        # STEP 4: OUTPUT (The Interface)
        robot.send_commands(target_angles, [0.0]*12)

        # STEP 5: TIMING (Maintain 500Hz)
        elapsed = time.perf_counter() - start_time
        if elapsed < DT:
            time.sleep(DT - elapsed)

if __name__ == "__main__":
    main()