import mujoco
import mujoco.viewer
import time
import numpy as np

# --- Configuration ---
GOAL_THRESHOLD = 0.3
MAX_STEPS = 2000
COLLISION_PENALTY = -10.0
GOAL_REWARD = 100.0
TIME_PENALTY = -0.01
DISTANCE_REWARD_SCALE = 1.0


def get_random_pos(min_val, max_val):
    """Returns a random [x, y] position within the given range."""
    return np.random.uniform(min_val, max_val, size=2)


def check_distance(pos1, pos2, min_dist):
    """Checks if the distance between two points is greater than min_dist."""
    return np.linalg.norm(np.array(pos1) - np.array(pos2)) > min_dist


def get_robot_yaw(data):
    """Extract yaw angle from robot's quaternion (qpos[3:7] = [w, x, y, z])."""
    w, x, y, z = data.qpos[3:7]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))


def get_goal_pos(model):
    """Get the goal's XY position."""
    goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'goal')
    return model.body_pos[goal_id][0:2].copy()


def get_goal_info(model, data):
    """Calculate distance and relative angle from robot to goal."""
    robot_pos = data.qpos[0:2]
    goal_pos = get_goal_pos(model)
    diff = goal_pos - robot_pos
    distance = np.linalg.norm(diff)

    world_angle = np.arctan2(diff[1], diff[0])
    robot_yaw = get_robot_yaw(data)
    rel_angle = (world_angle - robot_yaw + np.pi) % (2 * np.pi) - np.pi

    return distance, rel_angle


def get_observation(model, data):
    """Get observation vector: 36 LiDAR values + goal distance + goal relative angle."""
    lidar = data.sensordata[:36].copy()
    lidar[lidar < 0] = 5.0  # Replace -1 (no hit) with cutoff value

    goal_dist, goal_angle = get_goal_info(model, data)
    return np.concatenate([lidar, [goal_dist, goal_angle]])


def check_collision(model, data):
    """Check if robot chassis collides with walls or obstacles."""
    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'chassis')
    obstacle_names = [
        'wall_front', 'wall_back', 'wall_left', 'wall_right',
        'obstacle_1', 'obstacle_2', 'obstacle_3',
    ]
    obstacle_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in obstacle_names
    }

    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == chassis_id and c.geom2 in obstacle_ids) or \
           (c.geom2 == chassis_id and c.geom1 in obstacle_ids):
            return True
    return False


def check_done(model, data, step_count):
    """Check termination: goal reached, collision, or timeout."""
    goal_dist, _ = get_goal_info(model, data)
    collision = check_collision(model, data)
    goal_reached = goal_dist < GOAL_THRESHOLD
    timeout = step_count >= MAX_STEPS

    done = goal_reached or collision or timeout
    info = {
        'goal_reached': goal_reached,
        'collision': collision,
        'timeout': timeout,
        'goal_dist': goal_dist,
    }
    return done, info


def compute_reward(model, data, prev_dist, done_info):
    """Compute step reward with distance shaping + terminal bonuses."""
    goal_dist, _ = get_goal_info(model, data)
    reward = 0.0

    # Distance shaping: reward for getting closer, penalty for moving away
    reward += DISTANCE_REWARD_SCALE * (prev_dist - goal_dist)

    # Small time penalty to encourage efficiency
    reward += TIME_PENALTY

    # Terminal rewards
    if done_info['goal_reached']:
        reward += GOAL_REWARD
    if done_info['collision']:
        reward += COLLISION_PENALTY

    return reward, goal_dist


def reset_env(model, data):
    """Reset environment with randomized positions. Returns (obs, goal_dist)."""
    mujoco.mj_resetData(model, data)

    # 1. Randomize robot position
    robot_pos = get_random_pos(-1.5, 1.5)
    data.qpos[0:2] = robot_pos
    data.qpos[2] = 0.1
    data.qpos[3:7] = [1, 0, 0, 0]

    # Zero out velocities and actuator controls
    data.qvel[:] = 0
    data.ctrl[:] = 0

    # Sync start_point marker
    start_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'start_point')
    model.geom_pos[start_id][0:2] = robot_pos

    # 2. Randomize goal position (min 1.0m from robot)
    goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'goal')
    while True:
        goal_pos = get_random_pos(-2.2, 2.2)
        if check_distance(robot_pos, goal_pos, 1.0):
            model.body_pos[goal_id][0:2] = goal_pos
            break

    # 3. Randomize obstacles (no overlap with robot, goal, or each other)
    obstacles = ['obstacle_1', 'obstacle_2', 'obstacle_3']
    placed = []
    for obs_name in obstacles:
        obs_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obs_name)
        while True:
            obs_pos = get_random_pos(-2.0, 2.0)
            if not check_distance(obs_pos, robot_pos, 0.7):
                continue
            if not check_distance(obs_pos, goal_pos, 0.7):
                continue
            if any(not check_distance(obs_pos, p, 0.8) for p in placed):
                continue
            model.geom_pos[obs_id][0:2] = obs_pos
            placed.append(obs_pos)
            break

    mujoco.mj_forward(model, data)

    obs = get_observation(model, data)
    goal_dist, _ = get_goal_info(model, data)
    return obs, goal_dist


def main():
    model = mujoco.MjModel.from_xml_path("xml/diff_robot.xml")
    data = mujoco.MjData(model)

    obs, prev_dist = reset_env(model, data)
    step_count = 0
    total_reward = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Random control for demo (replace with RL policy)
            data.ctrl[0] = np.random.uniform(-5, 5)  # left motor
            data.ctrl[1] = np.random.uniform(-5, 5)  # right motor

            mujoco.mj_step(model, data)
            step_count += 1

            # Check termination
            done, done_info = check_done(model, data, step_count)

            # Compute reward
            reward, prev_dist = compute_reward(model, data, prev_dist, done_info)
            total_reward += reward

            # Get observation
            obs = get_observation(model, data)

            if done:
                reason = "GOAL!" if done_info['goal_reached'] else \
                         "COLLISION" if done_info['collision'] else "TIMEOUT"
                print(f"Episode done: {reason} | Steps: {step_count} | "
                      f"Total reward: {total_reward:.2f} | "
                      f"Final dist: {done_info['goal_dist']:.2f}")
                obs, prev_dist = reset_env(model, data)
                step_count = 0
                total_reward = 0.0

            viewer.sync()

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
