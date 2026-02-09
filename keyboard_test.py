"""
Keyboard-controlled test for the differential drive robot.

Controls:
  W / Up    : forward
  S / Down  : backward
  A / Left  : turn left (in-place)
  D / Right : turn right (in-place)
  Q         : curve left forward
  E         : curve right forward
  Space     : stop
  R         : reset
  ESC       : quit
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
import time

# GLFW key codes
KEY_W, KEY_A, KEY_S, KEY_D = 87, 65, 83, 68
KEY_R, KEY_Q, KEY_E = 82, 81, 69
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_SPACE = 32
KEY_ESC = 256

MOTOR_SPEED = 15.0

# --- Same constants as DiffRobotEnv ---
GOAL_THRESHOLD = 0.3
MAX_STEPS = 2000
N_SUBSTEPS = 10

current_ctrl = [0.0, 0.0]
should_reset = False
should_quit = False


def key_callback(key):
    global should_reset, should_quit

    S = MOTOR_SPEED

    actions = {
        KEY_W:     ( S,        S),
        KEY_UP:    ( S,        S),
        KEY_S:     (-S,       -S),
        KEY_DOWN:  (-S,       -S),
        KEY_A:     (-S * 0.6,  S * 0.6),
        KEY_LEFT:  (-S * 0.6,  S * 0.6),
        KEY_D:     ( S * 0.6, -S * 0.6),
        KEY_RIGHT: ( S * 0.6, -S * 0.6),
        KEY_Q:     ( S * 0.3,  S),
        KEY_E:     ( S,        S * 0.3),
        KEY_SPACE: ( 0.0,      0.0),
    }

    if key in actions:
        current_ctrl[0], current_ctrl[1] = actions[key]
    elif key == KEY_R:
        should_reset = True
    elif key == KEY_ESC:
        should_quit = True


def main():
    global should_reset, should_quit

    xml_path = os.path.join(os.path.dirname(__file__), "xml", "diff_robot.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    rng = np.random.default_rng()

    # --- Cache IDs (same as DiffRobotEnv) ---
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    robot_body_ids = set()
    for i in range(model.nbody):
        bid = i
        while bid != 0:
            if bid == base_body_id:
                robot_body_ids.add(i)
                break
            bid = model.body_parentid[bid]
    robot_body_ids.add(base_body_id)

    robot_geom_ids = {
        i for i in range(model.ngeom)
        if model.geom_bodyid[i] in robot_body_ids
    }
    obstacle_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in ["wall_front", "wall_back", "wall_left", "wall_right",
                   "obstacle_1", "obstacle_2", "obstacle_3"]
    }
    start_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "start_point")
    goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")
    obstacle_names = ["obstacle_1", "obstacle_2", "obstacle_3"]
    obstacle_mocap_ids = {}
    for name in obstacle_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        obstacle_mocap_ids[name] = model.body_mocapid[bid]

    def check_collision():
        for i in range(data.ncon):
            c = data.contact[i]
            if (c.geom1 in robot_geom_ids and c.geom2 in obstacle_ids) or \
               (c.geom2 in robot_geom_ids and c.geom1 in obstacle_ids):
                return True
        return False

    def goal_distance():
        return np.linalg.norm(
            model.body_pos[goal_body_id][0:2] - data.qpos[0:2]
        )

    def reset_env():
        current_ctrl[0] = current_ctrl[1] = 0.0
        mujoco.mj_resetData(model, data)

        # Randomize robot
        robot_pos = rng.uniform(-1.5, 1.5, size=2)
        data.qpos[0:2] = robot_pos
        data.qpos[2] = 0.1
        data.qpos[3:7] = [1, 0, 0, 0]

        model.geom_pos[start_id][0:2] = robot_pos

        # Randomize goal (min 1.2m from robot)
        while True:
            goal_pos = rng.uniform(-2.0, 2.0, size=2)
            if np.linalg.norm(goal_pos - robot_pos) > 1.2:
                model.body_pos[goal_body_id][0:2] = goal_pos
                break

        # Randomize obstacles
        placed = []
        for name in obstacle_names:
            mocap_id = obstacle_mocap_ids[name]
            while True:
                obs_pos = rng.uniform(-1.8, 1.8, size=2)
                if np.linalg.norm(obs_pos - robot_pos) < 1.0:
                    continue
                if np.linalg.norm(obs_pos - goal_pos) < 1.0:
                    continue
                if any(np.linalg.norm(obs_pos - p) < 1.2 for p in placed):
                    continue
                data.mocap_pos[mocap_id][0:2] = obs_pos
                placed.append(obs_pos)
                break

        mujoco.mj_forward(model, data)
        return 0  # step_count

    # --- Initial reset ---
    step_count = reset_env()
    print("[New episode]")

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    print(__doc__)

    dt = model.opt.timestep * N_SUBSTEPS
    try:
        while viewer.is_running() and not should_quit:
            if should_reset:
                should_reset = False
                step_count = reset_env()
                print("[Manual reset]")
                continue

            data.ctrl[0] = current_ctrl[0]
            data.ctrl[1] = current_ctrl[1]

            # Step physics (N_SUBSTEPS like the env)
            collision = False
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(model, data)
                if not collision and check_collision():
                    collision = True
            step_count += 1

            dist = goal_distance()
            goal_reached = dist < GOAL_THRESHOLD
            timeout = step_count >= MAX_STEPS

            # Auto-reset on termination
            if collision or goal_reached or timeout:
                reason = "Collision!" if collision else \
                         "Goal reached!" if goal_reached else "Timeout"
                print(f"[{reason}] steps={step_count}, dist={dist:.2f}")
                time.sleep(1.0)  # pause briefly so user can see result
                step_count = reset_env()
                print("[New episode]")

            viewer.sync()
            time.sleep(dt)
    finally:
        viewer.close()
        print("Done.")


if __name__ == "__main__":
    main()
