
https://github.com/user-attachments/assets/027660f4-75a6-4265-8bf1-c44d5cf3b1c5

```sh
uv venv --python $(brew --prefix python@3.12)/bin/python3.12
uv sync
uv run mjpython main.py
```

```sh
uv run python train.py train --total-timesteps 1000000
uv run mjpython train.py enjoy --model-path ./models/ppo_diff_robot.zip
```
