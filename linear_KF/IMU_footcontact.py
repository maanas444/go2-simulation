import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
steps = 300

true_pos = 0
true_vel = 0
true_acc = 0.2

acc_noise_std = 0.2

# State vector: [position, velocity]
x = np.array([[0],   
              [0]])  

F = np.array([[1, dt],
              [0, 1]])

B = np.array([[0.5 * dt**2],
              [dt]])

Q = np.array([[0.05, 0],
              [0, 0.05]])

P = np.eye(2)

true_positions = []
imu_positions = []
kf_positions = []

imu_pos = 0
imu_vel = 0

for step in range(steps):
    # --- 1. SIMULATE TRUE STATE ---
    true_vel += true_acc * dt
    true_pos += true_vel * dt
    true_positions.append(true_pos)

    # --- 2. SIMULATE IMU DRIFT ---
    a_meas = true_acc + np.random.randn() * acc_noise_std
    imu_vel += a_meas * dt
    imu_pos += imu_vel * dt
    imu_positions.append(imu_pos)

    # Simulate stance (foot on ground) vs swing (foot in air)
    contact = (step % 30) < 15  

    # --- 3. KALMAN FILTER PREDICT STEP (IMU) ---
    x = F @ x + B * a_meas
    P = F @ P @ F.T + Q

    # --- 4. KALMAN FILTER UPDATE STEP (FOOT CONTACT/KINEMATICS) ---
    # Only correct the estimate when the foot is firmly on the ground
    if contact and len(kf_positions) > 0:
        # In a real robot, this zc comes from Forward Kinematics
        zc = np.array([[kf_positions[-1]]]) 
        Hc = np.array([[1, 0]])
        Rc = np.array([[0.5]])  # Confidence in our foot placement

        y = zc - (Hc @ x)
        S = Hc @ P @ Hc.T + Rc
        K = P @ Hc.T @ np.linalg.inv(S)

        x = x + K @ y
        P = (np.eye(2) - K @ Hc) @ P

    kf_positions.append(x[0, 0])

# --- 5. PLOTTING ---
plt.figure(figsize=(10, 6))

plt.plot(true_positions, label="True Position", linewidth=2)
plt.plot(imu_positions, label="IMU Integrated (Drift)", linestyle="dashed")
plt.plot(kf_positions, label="KF (IMU + Foot Contact)", linewidth=2)

plt.legend()
plt.title("Proprioceptive Sensor Fusion: IMU + Foot Contact")
plt.xlabel("Time Step")
plt.ylabel("Position")

plt.savefig("fusion_proprioception_only.png")
print("Saved fusion_proprioception_only.png")