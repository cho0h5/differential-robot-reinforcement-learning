import mujoco
import mujoco.viewer
import time
import numpy as np

def get_random_pos(min_val, max_val):
    """Returns a random [x, y] position within the given range."""
    return np.random.uniform(min_val, max_val, size=2)

def check_distance(pos1, pos2, min_dist):
    """Checks if the distance between two points is greater than min_dist."""
    return np.linalg.norm(np.array(pos1) - np.array(pos2)) > min_dist

def reset_env(model, data):
    # 1. Randomize Robot and Start Point position
    # The start_point (marker) and robot should be at the same location.
    robot_pos = get_random_pos(-1.5, 1.5)
    
    # Set robot position (qpos[0:2] for X, Y)
    data.qpos[0:2] = robot_pos
    data.qpos[2] = 0.1  # Fixed height
    data.qpos[3:7] = [1, 0, 0, 0] # Reset orientation (Quaternion)

    # Sync start_point marker with robot position
    start_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'start_point')
    model.geom_pos[start_id][0:2] = robot_pos

    # 2. Randomize Goal position (Ensure it's not too close to the robot)
    goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'goal')
    while True:
        goal_pos = get_random_pos(-2.2, 2.2)
        if check_distance(robot_pos, goal_pos, 1.0): # Min 1.0m distance from robot
            model.body_pos[goal_id][0:2] = goal_pos
            break

    # 3. Randomize Obstacles (Ensure no overlap with robot or goal)
    obstacles = ['obstacle_1', 'obstacle_2', 'obstacle_3']
    for obs_name in obstacles:
        obs_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obs_name)
        while True:
            obs_pos = get_random_pos(-2.0, 2.0)
            # Check distance against robot and goal
            if check_distance(obs_pos, robot_pos, 0.7) and check_distance(obs_pos, goal_pos, 0.7):
                model.geom_pos[obs_id][0:2] = obs_pos
                break

    # Finalize position changes in physics engine
    mujoco.mj_forward(model, data)

def main():
    model_path = "xml/diff_robot.xml"

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    reset_env(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
