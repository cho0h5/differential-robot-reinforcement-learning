import argparse
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from diff_robot_env import DiffRobotEnv


def make_env():
    def _init():
        return DiffRobotEnv()
    return _init


def train(args):
    # Parallel envs for faster training (headless)
    train_env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])

    # Separate eval env (single)
    eval_env = DiffRobotEnv()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/eval/",
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
    )

    print(f"Training PPO for {args.total_timesteps} steps "
          f"with {args.n_envs} parallel envs...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save("./models/ppo_diff_robot")
    print("Training complete. Model saved to ./models/ppo_diff_robot.zip")

    train_env.close()
    eval_env.close()


def enjoy(args):
    """Load trained model and visualize."""
    env = DiffRobotEnv(render_mode="human")
    model = PPO.load(args.model_path)

    step_dt = env.model.opt.timestep * env.N_SUBSTEPS  # real-time step duration

    obs, _ = env.reset()
    total_reward = 0.0
    while True:
        step_start = time.time()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Real-time sync
        elapsed = time.time() - step_start
        if elapsed < step_dt:
            time.sleep(step_dt - elapsed)

        if terminated or truncated:
            reason = "GOAL!" if info["goal_reached"] else \
                     "COLLISION" if info["collision"] else "TIMEOUT"
            print(f"{reason} | Reward: {total_reward:.2f} | "
                  f"Dist: {info['goal_dist']:.2f}")
            obs, _ = env.reset()
            total_reward = 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--total-timesteps", type=int, default=1_000_000)
    p_train.add_argument("--n-envs", type=int, default=4)
    p_train.add_argument("--eval-freq", type=int, default=10_000)

    # enjoy (visualize trained model)
    p_enjoy = sub.add_parser("enjoy")
    p_enjoy.add_argument("--model-path", type=str,
                         default="./models/ppo_diff_robot.zip")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "enjoy":
        enjoy(args)
    else:
        parser.print_help()
