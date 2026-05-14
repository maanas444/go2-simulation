import numpy as np

class StateEstimator:
    def __init__(self, dt: float):
        self.dt = dt
        self.x = np.array([[0.28], [0.0]])
        self.P = np.eye(2) * 0.5
        self.Q = np.diag([0.001, 0.001])
        self.R = 0.01
        
    def predict(self):
        a_world = z_accel - 9.81
        F = np.array([[1, self.dt], [0, 1]])
        B = np.array([[0.5 * self.dt**2], [self.dt]])
        
        self.x = F @ self.x + B * a_world
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, z_pos):
        H = np.array([[1, 0]])
        y = measured_z - (H @ self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S
        self.x = self.x + K * y
        self.P = (np.eye(2) - K @ H) @ self.P
        
    @property
    def height(self) -> float:
        return float(self.x[0][0])

    @property
    def vertical_velocity(self) -> float:
        return float(self.x[1][0])