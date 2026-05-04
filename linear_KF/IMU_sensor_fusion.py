import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
steps = 300

true_pos = 0
true_vel = 0
true_acc = 0.2

acc_noise_std = 0.2
vision_noise_std = 2.0

x = np.array([[0],   # position
              [0]])  # velocity

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
vision_measurements = []

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

    contact = (step % 30) < 15  # stance vs swing

    x = F @ x + B * a_meas
    P = F @ P @ F.T + Q

    if contact and len(kf_positions) > 0:
        zc = np.array([[kf_positions[-1]]])
        Hc = np.array([[1, 0]])
        Rc = np.array([[0.5]])  # moderate confidence

        y = zc - (Hc @ x)
        S = Hc @ P @ Hc.T + Rc
        K = P @ Hc.T @ np.linalg.inv(S)

        x = x + K @ y
        P = (np.eye(2) - K @ Hc) @ P

    if step % 10 == 0:
        zv = true_pos + np.random.randn() * vision_noise_std
        vision_measurements.append((step, zv))

        Hv = np.array([[1, 0]])
        Rv = np.array([[vision_noise_std**2]])

        y = zv - (Hv @ x)
        S = Hv @ P @ Hv.T + Rv
        K = P @ Hv.T @ np.linalg.inv(S)

        x = x + K @ y
        P = (np.eye(2) - K @ Hv) @ P

    kf_positions.append(x[0, 0])

plt.figure(figsize=(10, 6))

plt.plot(true_positions, label="True Position", linewidth=2)
plt.plot(imu_positions, label="IMU Integrated (Drift)", linestyle="dashed")
plt.plot(kf_positions, label="KF (IMU + Contact + Vision)", linewidth=2)

if vision_measurements:
    steps_v, vals_v = zip(*vision_measurements)
    plt.scatter(steps_v, vals_v, label="Vision (SLAM)", alpha=0.6)

plt.legend()
plt.title("Sensor Fusion: IMU + Foot Contact + Vision")
plt.xlabel("Time Step")
plt.ylabel("Position")

plt.savefig("fusion_full.png")
print("Saved fusion_full.png")