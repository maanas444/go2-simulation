import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
steps = 100

true_pos = 0
true_vel = 1.0

process_noise_std = 0.1
measurement_noise_std = 1.0

F = np.array([[1, dt], [0, 1]])
H = np.array([[1, 0]])
Q = np.array([[0.01, 0], [0, 0.01]])
R = np.array([[measurement_noise_std**2]])

x = np.array([[0], [0]])

P = np.eye(2)

true_positions = []
measurements = []
estimates = []

for step in range(steps):
    true_pos += true_vel * dt
    true_positions.append(true_pos)

    # Single noisy measurement
    z = true_pos + np.random.randn() * measurement_noise_std
    measurements.append(z)

    # Predict
    x = F @ x
    P = F @ P @ F.T + Q

    # Update
    y = z - (H @ x)[0]
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    x = x + K * y
    P = (np.eye(2) - K @ H) @ P

    estimates.append(x[0, 0])

plt.figure()
plt.plot(true_positions, label="True Position")
plt.plot(measurements, label="Measurements", linestyle="dotted")
plt.plot(estimates, label="KF Estimate")
plt.legend()
plt.title("1D Kalman Filter")
plt.savefig("kf_output.png")
print("Saved plot to kf_output.png")