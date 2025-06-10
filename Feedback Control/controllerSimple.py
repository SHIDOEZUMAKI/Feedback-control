import numpy as np
# import KCalculate
# from KCalculate import calculate_lqr_gain

def normalize_angle(angle):
    """
    Normalize an angle to the range [-pi, pi].

    Parameters:
        angle (float): The angle in radians.

    Returns:
        float: The normalized angle in radians.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def controller(state, target_pos, dt):
    """
    LQR controller.

    This function computes the error between the current state and target,
    applies a precomputed LQR gain matrix K to determine desired world-frame
    velocities, and transforms them into the body frame.

    Parameters:
        state: list [x, y, z, roll, pitch, yaw], units are meters and radians.
        target_pos: tuple (x, y, z, yaw), target position and heading.
        dt: time step (seconds) — not used directly in this LQR model.

    Returns:
        tuple: (ux, uy, uz, w_yaw) desired velocities in the body frame.
    """
    # Extract state variables
    x, y, z, roll, pitch, yaw = state
    target_x, target_y, target_z, target_yaw = target_pos

    # --- 1. Compute the error state vector ---
    # We track errors in x, y, z positions and yaw angle.
    error = np.array([
        (x - target_x),
        (y - target_y),
        z - target_z,
        normalize_angle(yaw - target_yaw),
    ])

    # --- 3. Calculate LQR control input U = -K * X_error ---
    # U_desired = [vx_des, vy_des, vz_des, wyaw_des] (World frame)
    # K is the LQR gain matrix.
    # This matrix is pre-calculated to reduce the amount of
    # computation for each function call.
    # If KCalculate.py and calculate_lqr_gain were available,
    # K could be dynamically calculated using:
    # K, _ = calculate_lqr_gain()
    K = np.array([[0.45511111, 0.        , 0.        , 0.        ],
                  [0.        , 0.45511111, 0.        , 0.        ],
                  [0.        , 0.        , 4.83333333, 0.        ],
                  [0.        , 0.        , 0.        , 9.38596491]])

    U_desired = -K @ error

    vx_des, vy_des, vz_des, wyaw_des = U_desired # Desired velocities in the world frame

    # --- 3. Transform desired velocities from world frame to body frame ---
    # ux: desired velocity along the body x-axis
    # uy: desired velocity along the body y-axis
    # uz: desired velocity along the body z-axis (same as world z-axis for this transformation)
    # w_yaw: desired yaw rate (same as world yaw rate)
    ux = vx_des * np.cos(yaw) + vy_des * np.sin(yaw)
    uy = -vx_des * np.sin(yaw) + vy_des * np.cos(yaw)
    uz = vz_des
    w_yaw = wyaw_des

    # Optional: Add velocity limits.
    # These are commented out because they are expected to be implemented in run.py.
    # max_xy_speed = ... # Define maximum speed in x-y plane
    # max_z_speed = ...  # Define maximum speed in z direction
    # max_yaw_rate = ... # Define maximum yaw rate
    # ux = np.clip(ux, -max_xy_speed, max_xy_speed)
    # uy = np.clip(uy, -max_xy_speed, max_xy_speed)
    # uz = np.clip(uz, -max_z_speed, max_z_speed)
    # w_yaw = np.clip(w_yaw, -max_yaw_rate, max_yaw_rate)

    # Control command tuple in the body frame
    u = (ux, uy, uz, w_yaw)
    return u


""" KCalculate.py
This file is showing how to calculate the LQR gain matrix K.
This is not required for the controller to work, but it can be used to
visualize the LQR gain matrix K.
And it also contain some definition and assumption about controller model.

import numpy as np
from scipy.linalg import solve_discrete_are

def calculate_lqr_gain():

    # Compute the optimal LQR gain matrix K and cost-to-go matrix P for a
    # discrete-time integrator model:
    # 
    #     e_{k+1} = A e_k + B u_k,
    # where A = 0 (all zeros) and B = I (identity).
    # 
    # The cost function is:
    #     J = sum{ e_k^T Q e_k + u_k^T R u_k }.
    # 
    # Returns:
    #     K (ndarray): Gain matrix shape (4,4).
    #     P (ndarray): Solution of Discrete-time Algebraic Riccati Eq (DARE).
    # Solve the discrete-time Algebraic Riccati Equation (DARE) and compute
    # the Linear Quadratic Regulator (LQR) gain matrix K for a simple integrator model.
    # 
    # The system dynamics follow the error model: e_{k+1} = e_k + u_k,
    # equivalent to x_{k+1} = A x_k + B u_k with A = 0, B = I.
    # 
    # Returns:
    #     K (ndarray): LQR gain matrix of shape (4, 4).
    #     P (ndarray): Solution to the DARE (cost-to-go matrix).

    # System matrices for integrator model
    A = np.zeros((4, 4)) # No inherent dynamics: next error = current error + input
    B = np.eye(4) # Control acts directly on each error channel

    # Cost weighting matrices
    Q = np.diag([5.12, 5.12, 5.8, 5.35]) # State error weights
    R = np.diag([11.25, 11.25, 1.2, 0.57]) # Control effort weights

    # Solve the discrete-time Algebraic Riccati Equation: A' P A - P - (A' P B)
    #                                           * (R + B' P B)^{-1} * (B' P A) + Q = 0
    P = solve_discrete_are(A, B, Q, R)

    # Compute LQR gain: K = (R + B' P B)^{-1} (B' P A)
    # For this model, A = 0 simplifies the expression to K = R^{-1} B^T P
    K = np.linalg.inv(R) @ B.T @ P

    return K, P

if __name__ == "__main__":
    K, P = calculate_lqr_gain()
    print("LQR Gain K:")
    print(K)
    print("\nDARE Solution P:")
    print(P)
"""
