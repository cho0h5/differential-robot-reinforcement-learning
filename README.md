```sh
uv venv --python $(brew --prefix python@3.12)/bin/python3.12
uv sync
uv run mjpython main.py
```