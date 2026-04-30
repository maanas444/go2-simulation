#𝑣𝑘=𝑣𝑘−1+𝑎⋅Δ𝑡   -> new velocity = old velocity + acceleration over time
#𝑥𝑘=𝑥𝑘−1+𝑣𝑘⋅Δ𝑡  -> new postion = old position + delta distance in timestep

import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
steps = 200

true_pos = 0
true_vel = 0
true_acc = 0.1

acc_noise_std = 0.2
pos_noise_std = 2.0

true_positions = []
imu_positions = []
kf_positions = []

x = np.array([[0], [0]])
F = np.array([[1, dt], [0, 1]])
B = np.array([[0.5 * dt**2], [dt]])
H = np.array([[1, 0]])
Q = np.array([[0.1, 0], [0, 0.1]])
R = np.array([[pos_noise_std**2]])

P = np.eye(2)

imu_pos = 0
imu_vel = 0

for step in range(steps):
    true_vel += true_acc * dt
    true_pos += true_vel * dt
    true_positions.append(true_pos)

    a_meas = true_acc + np.random.randn() * acc_noise_std
    
    imu_vel += a_meas * dt
    imu_pos += imu_vel * dt
    imu_positions.append(imu_pos)
    
    z = true_pos + np.random.randn() * pos_noise_std
    
    x = F @ x + B * a_meas
    P = F @ P @ F.T + Q

    y = z - (H @ x)[0]
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    x = x + K * y
    P = (np.eye(2) - K @ H) @ P

    kf_positions.append(x[0, 0])

# Plot
plt.figure()
plt.plot(true_positions, label="True Position")
plt.plot(imu_positions, label="IMU Integrated (Drift)", linestyle="dashed")
plt.plot(kf_positions, label="KF Estimate")
plt.legend()
plt.title("IMU Drift + Kalman Filter Correction")

plt.savefig("imu_kf.png")
print("Saved plot to imu_kf.png")