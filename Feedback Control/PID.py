import numpy as np
import time
from src.PID_controller import PIDController

# Global variables for state estimation using fixed dt and filtering
last_pos = None
last_filtered_vel = None
last_filtered_acc = None
stable_counter = 0

# Filtering coefficients
alpha_vel = 0.2
alpha_acc = 0.2

# Conservative PID gains for stability without oscillation
pid_pos = PIDController(0.8, 0.0, 0.02, [0.3, 0.3, 0.3])
pid_vel = PIDController(0.4, 0.005, 0.01, [0.2, 0.2, 0.2])
pid_acc = PIDController(0.15, 0.0, 0.003, [0.1, 0.1, 0.1])
pid_yaw = PIDController(0.8, 0.0, 0.02, [0.3, 0.3, 0.3])

# Thresholds
stable_counter = 0
STABLE_REQUIRED_STEPS = 20
ultra_freeze_zone = 0.005  # 5mm precision freeze
yaw_threshold = 0.03
output_epsilon = 0.02


def controller(state, target, dt):
    global last_pos, last_filtered_vel, last_filtered_acc, stable_counter

    pos = np.array(state[0:3])
    yaw = state[5]
    target_pos = np.array(target[0:3])
    target_yaw = target[3]

    if last_pos is None:
        raw_vel = np.zeros(3)
        filtered_vel = np.zeros(3)
        filtered_acc = np.zeros(3)
    else:
        raw_vel = (pos - last_pos) / dt
        filtered_vel = alpha_vel * raw_vel + (1 - alpha_vel) * last_filtered_vel
        raw_acc = (filtered_vel - last_filtered_vel) / dt
        filtered_acc = alpha_acc * raw_acc + (1 - alpha_acc) * last_filtered_acc

    pos_error = target_pos - pos
    yaw_error = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
    
    print(
        f"[误差] x: {pos_error[0]:.3f} m, y: {pos_error[1]:.3f} m, "
        f"z: {pos_error[2]:.3f} m, yaw: {yaw_error:.3f} rad"
    )

    u_pos = pid_pos.control_update(pos_error, dt)
    vel_error = u_pos - filtered_vel
    u_vel = pid_vel.control_update(vel_error, dt)
    acc_error = u_vel - filtered_acc
    u_acc = pid_acc.control_update(acc_error, dt)
    v_command = u_pos + 0.2 * u_acc

    # Convert to body frame
    R = np.array([[np.cos(yaw), np.sin(yaw)],
                  [-np.sin(yaw), np.cos(yaw)]])
    v_xy_body = R @ v_command[0:2]
    v_command_transformed = np.array([v_xy_body[0], v_xy_body[1], v_command[2]])

    yaw_rate_output = pid_yaw.control_update(np.array([yaw_error, 0, 0]), dt)
    yaw_rate_command = np.clip(yaw_rate_output[0], -0.5, 0.5)

    # Check freeze condition
    if (np.linalg.norm(pos_error) < ultra_freeze_zone and
        abs(yaw_error) < yaw_threshold and
        np.linalg.norm(v_command_transformed) < output_epsilon):
        stable_counter += 1
    else:
        stable_counter = 0

    if stable_counter >= STABLE_REQUIRED_STEPS:
        return (0.0, 0.0, 0.0, 0.0)

    last_pos = pos.copy()
    last_filtered_vel = filtered_vel.copy()
    last_filtered_acc = filtered_acc.copy()

    vx = np.clip(v_command_transformed[0], -0.5, 0.5)
    vy = np.clip(v_command_transformed[1], -0.5, 0.5)
    vz = np.clip(v_command_transformed[2], -0.4, 0.4)
    print(pos_error)
    return (vx, vy, vz, yaw_rate_command)