import numpy as np

def solve_continuous_are_hamiltonian(A, B, Q, R):
    """
    Solve the continuous-time Algebraic Riccati Equation (ARE) using the Hamiltonian matrix approach.
    """
    n = A.shape[0]
    R_inv = np.linalg.inv(R)
    Z = np.zeros((n, n))
    I = np.eye(n)

    # Hamiltonian matrix H
    H = np.block([
        [A, -B @ R_inv @ B.T],
        [-Q, -A.T]
    ])

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eig(H)

    # Select stable eigenvectors (Re(lambda) < 0)
    select = np.where(np.real(eigvals) < 0)[0]
    eigvecs = eigvecs[:, select]

    # Separate X and Y from eigvectors: eigvecs = [X; Y]
    X = eigvecs[:n, :]
    Y = eigvecs[n:, :]

    # Solve for P = Y @ X^{-1}
    P = np.real(Y @ np.linalg.inv(X))
    return P

def lqr(A, B, Q, R):
    """
    Compute the optimal feedback gain K for continuous-time LQR.
    Solve the Algebraic Riccati Equation (ARE):
      A^T P + P A - P B R^{-1} B^T P + Q = 0
    Then calculate K = R^{-1} B^T P
    """
    # Solve the Algebraic Riccati Equation to get P
    P = solve_continuous_are_hamiltonian(A, B, Q, R)
    # Compute the optimal feedback gain matrix
    K = np.linalg.inv(R) @ B.T @ P
    return K, P

def controller(state, target_pos, dt):
    """
    LQR Controller:

    Parameters:
      state: list [x, y, z, roll, pitch, yaw], units in meters and radians
      target_pos: tuple (x, y, z, yaw), target position and heading
      dt: time step (seconds) — not directly used in this model

    Returns:
      Velocity command tuple (velocity_x_setpoint, velocity_y_setpoint, velocity_z_setpoint, yaw_rate_setpoint)
    """
    # Extract state variables
    x, y, z, roll, pitch, yaw = state
    target_x, target_y, target_z, target_yaw = target_pos

    # Define state error
    error = np.array([
        (x - target_x) * np.cos(yaw) + (y - target_y) * np.sin(yaw),
        (y - target_y) * np.cos(yaw) - (x - target_x) * np.sin(yaw),
        z - target_z,
        (yaw - target_yaw + np.pi) % (2 * np.pi) - np.pi
    ])

    # # Set deadzone (e.g., consider as reached if within 0.1 m)
    # deadzone_pos = 0.01
    # deadzone_yaw = 0.5
    #
    # if abs(error[0]) < deadzone_pos:
    #     error[0] = 0
    # if abs(error[1]) < deadzone_pos:
    #     error[1] = 0
    # if abs(error[2]) < deadzone_pos:
    #     error[2] = 0
    # if abs(error[3]) < deadzone_yaw:
    #     error[3] = 0

    # For a simple integrator model: error dynamics are dot(e) = u,
    # State matrix A is a zero matrix, control matrix B is an identity matrix
    A = np.zeros((4, 4))
    B = np.eye(4)

    # Define the cost function weight matrices Q and R
    # Q penalizes state error, R penalizes control input
    # The values reflect a trade-off between accuracy and energy consumption
    Q = np.diag([3, 5.0, 10.0, 1.0])
    R = np.diag([10.0, 30.5, 1.0, 1.0])

    # Compute the optimal feedback gain matrix K
    K, _ = lqr(A, B, Q, R)
    
    # u elements correspond to (velocity_x_setpoint, velocity_y_setpoint, velocity_z_setpoint, yaw_rate_setpoint)
    u = -K @ error
    print("LQR error:", error)
    return tuple(u)

