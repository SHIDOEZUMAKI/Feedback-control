import numpy as np
from scipy.linalg import solve_discrete_are

def calculate_lqr_gain():
    """
    Solve the discrete-time Algebraic Riccati Equation (DARE) and compute
    the Linear Quadratic Regulator (LQR) gain matrix K for a simple integrator model.

    The system dynamics follow the error model: e_{k+1} = e_k + u_k,
    equivalent to x_{k+1} = A x_k + B u_k with A = 0, B = I.

    Returns:
        K (ndarray): LQR gain matrix of shape (4, 4).
        P (ndarray): Solution to the DARE (cost-to-go matrix).
    """
    # State transition matrix A (all zeros for integrator model)
    A = np.zeros((4, 4))
    # Control input matrix B (identity for direct control)
    B = np.eye(4)

    # Cost weighting matrices
    # Q penalizes state error, reflecting desired regulation performance
    Q = np.diag([5.12, 5.12, 5.8, 5.35])
    # R penalizes control effort, trading off energy consumption
    R = np.diag([11.25, 11.25, 1.2, 0.57])

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
