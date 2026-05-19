import numpy as np

class StateEstimator:
    def __init__(self, dt: float):
        self.dt = dt
        # State: [roll, pitch, yaw, z, z_dot]
        self.x = np.array([0.0, 0.0, 0.0, 0.28, 0.0])
        
        # Covariance Matrix (Uncertainty)
        self.P = np.eye(5) * 0.1
        
        # Process Noise (Physics uncertainty)
        self.Q = np.diag([0.001, 0.001, 0.001, 0.001, 0.01])
        
        # Measurement Noise (Trust in sensors)
        self.R = np.eye(3) * 0.05 

    def get_rotation_matrix(self, r, p, y):
        """Calculates the Body-to-World rotation matrix."""
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)

        # Standard R_zyx rotation matrix
        return np.array([
            [cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
            [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
            [-sp,   sr*cp,            cr*cp]
        ])

    def predict(self, gyro, accel):
        """
        Prediction step: Uses high-speed IMU data to project the state forward.
        Stability Aware: Rotates gravity to the world frame to isolate true acceleration.
        """
        r, p, y, z, z_dot = self.x
        
        # 1. Update orientation using Gyroscope
        self.x[0] += gyro[0] * self.dt
        self.x[1] += gyro[1] * self.dt
        self.x[2] += gyro[2] * self.dt
        
        # 2. Extract vertical acceleration in the world frame
        Rot = self.get_rotation_matrix(self.x[0], self.x[1], self.x[2])
        accel_world = Rot @ accel
        a_z = accel_world[2] - 9.81  # Remove gravity constant
        
        # 3. Update Height and Vertical Velocity
        self.x[3] += z_dot * self.dt + 0.5 * a_z * self.dt**2
        self.x[4] += a_z * self.dt
        
        # 4. Jacobian F - Linearizes the motion model
        # Simplification: assuming orientation changes are small between steps
        F = np.eye(5)
        F[3, 4] = self.dt 
        
        self.P = F @ self.P @ F.T + self.Q

    def update(self, foot_forces, measured_z, measured_roll, measured_pitch):
        """
        Update step: Fuses leg kinematics with the IMU guess.
        Sensor Fusion: Only updates if at least one foot is in contact.
        """
        # Threshold for foot contact (e.g., 40 Newtons)
        contacts = [f > 40 for f in foot_forces]
        
        if not any(contacts):
            return # Airborne: ignore kinematics to prevent drift

        # Measurement vector: [roll, pitch, height]
        z_vec = np.array([measured_roll, measured_pitch, measured_z])
        
        # Observation Matrix H (Mapping state to measurements)
        H = np.zeros((3, 5))
        H[0, 0] = 1 # Roll
        H[1, 1] = 1 # Pitch
        H[2, 3] = 1 # Height
        
        # Kalman Math
        y = z_vec - (H @ self.x)  # Innovation (Error)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S) # Kalman Gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ H) @ self.P

    @property
    def orientation(self):
        return self.x[0], self.x[1], self.x[2] # Roll, Pitch, Yaw

    @property
    def height(self):
        return self.x[3]