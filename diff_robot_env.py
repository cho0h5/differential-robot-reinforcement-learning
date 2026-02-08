import os
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces


class DiffRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    # --- Configuration ---
    GOAL_THRESHOLD = 0.3
    MAX_STEPS = 2000
    COLLISION_PENALTY = -10.0
    GOAL_REWARD = 100.0
    TIME_PENALTY = -0.01
    DISTANCE_REWARD_SCALE = 1.0
    LIDAR_CUTOFF = 5.0
    N_LIDAR = 36
    N_SUBSTEPS = 10  # physics steps per env step (0.002 * 10 = 0.02s = 50Hz control)
    ACTION_SCALE = 20.0  # map [-1, 1] → [-20, 20] motor command

    def __init__(self, render_mode=None):
        super().__init__()
        xml_path = os.path.join(os.path.dirname(__file__), "xml", "diff_robot.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Action: [left_motor, right_motor] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation: 36 LiDAR (0~1) + goal_dist (0~1) + goal_angle (-1~1) = 38
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.N_LIDAR + 2,), dtype=np.float32
        )

        # Cache geom/body IDs
        # Collect ALL robot geom IDs (chassis, wheels, casters, front_mark)
        base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        robot_body_ids = set()
        for i in range(self.model.nbody):
            bid = i
            while bid != 0:
                if bid == base_body_id:
                    robot_body_ids.add(i)
                    break
                bid = self.model.body_parentid[bid]
        robot_body_ids.add(base_body_id)

        self._robot_geom_ids = {
            i for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] in robot_body_ids
        }
        self._obstacle_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in [
                "wall_front", "wall_back", "wall_left", "wall_right",
                "obstacle_1", "obstacle_2", "obstacle_3",
            ]
        }
        self._start_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "start_point"
        )
        self._goal_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )
        self._obstacle_geom_names = ["obstacle_1", "obstacle_2", "obstacle_3"]

        self.step_count = 0
        self.prev_dist = 0.0

        # Rendering
        self.render_mode = render_mode
        self.viewer = None

    # ---- Gymnasium API ----

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize robot position
        robot_pos = self.np_random.uniform(-1.5, 1.5, size=2)
        self.data.qpos[0:2] = robot_pos
        self.data.qpos[2] = 0.1
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0

        # Sync start marker
        self.model.geom_pos[self._start_id][0:2] = robot_pos

        # Randomize goal (min 1.0m from robot)
        while True:
            goal_pos = self.np_random.uniform(-2.2, 2.2, size=2)
            if np.linalg.norm(goal_pos - robot_pos) > 1.0:
                self.model.body_pos[self._goal_body_id][0:2] = goal_pos
                break

        # Randomize obstacles (no overlap with robot, goal, or each other)
        placed = []
        for obs_name in self._obstacle_geom_names:
            obs_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, obs_name
            )
            while True:
                obs_pos = self.np_random.uniform(-2.0, 2.0, size=2)
                if np.linalg.norm(obs_pos - robot_pos) < 0.7:
                    continue
                if np.linalg.norm(obs_pos - goal_pos) < 0.7:
                    continue
                if any(np.linalg.norm(obs_pos - p) < 0.8 for p in placed):
                    continue
                self.model.geom_pos[obs_id][0:2] = obs_pos
                placed.append(obs_pos)
                break

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_dist = self._goal_distance()
        obs = self._get_obs()
        return obs, {"goal_dist": self.prev_dist}

    def step(self, action):
        # Scale action [-1, 1] → [-20, 20] and apply
        self.data.ctrl[0] = float(action[0]) * self.ACTION_SCALE
        self.data.ctrl[1] = float(action[1]) * self.ACTION_SCALE

        # Step physics N times, check collision at every substep
        collision = False
        for _ in range(self.N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            if not collision and self._check_collision():
                collision = True
        self.step_count += 1

        # Compute obs, reward, done
        obs = self._get_obs()
        goal_dist = self._goal_distance()
        goal_reached = goal_dist < self.GOAL_THRESHOLD
        timeout = self.step_count >= self.MAX_STEPS

        # Reward
        reward = self.DISTANCE_REWARD_SCALE * (self.prev_dist - goal_dist)
        reward += self.TIME_PENALTY
        if goal_reached:
            reward += self.GOAL_REWARD
        if collision:
            reward += self.COLLISION_PENALTY
        self.prev_dist = goal_dist

        # Gymnasium uses terminated (task end) vs truncated (time limit)
        terminated = goal_reached or collision
        truncated = timeout

        info = {
            "goal_dist": goal_dist,
            "goal_reached": goal_reached,
            "collision": collision,
            "timeout": timeout,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ---- Internal helpers ----

    def _get_robot_yaw(self):
        w, x, y, z = self.data.qpos[3:7]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    def _goal_distance(self):
        robot_pos = self.data.qpos[0:2]
        goal_pos = self.model.body_pos[self._goal_body_id][0:2]
        return np.linalg.norm(goal_pos - robot_pos)

    def _goal_relative_angle(self):
        robot_pos = self.data.qpos[0:2]
        goal_pos = self.model.body_pos[self._goal_body_id][0:2]
        diff = goal_pos - robot_pos
        world_angle = np.arctan2(diff[1], diff[0])
        rel_angle = (world_angle - self._get_robot_yaw() + np.pi) % (2 * np.pi) - np.pi
        return rel_angle

    def _get_obs(self):
        # LiDAR: normalize to [0, 1]
        lidar = self.data.sensordata[:self.N_LIDAR].copy()
        lidar[lidar < 0] = self.LIDAR_CUTOFF
        lidar_norm = lidar / self.LIDAR_CUTOFF  # [0, 1]

        # Goal distance: normalize to ~[0, 1] (max ~7m diagonal, use cutoff)
        goal_dist_norm = min(self._goal_distance() / self.LIDAR_CUTOFF, 1.0)

        # Goal angle: normalize to [-1, 1]
        goal_angle_norm = self._goal_relative_angle() / np.pi

        obs = np.concatenate([lidar_norm, [goal_dist_norm, goal_angle_norm]])
        return obs.astype(np.float32)

    def _check_collision(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in self._robot_geom_ids and c.geom2 in self._obstacle_ids) or \
               (c.geom2 in self._robot_geom_ids and c.geom1 in self._obstacle_ids):
                return True
        return False
