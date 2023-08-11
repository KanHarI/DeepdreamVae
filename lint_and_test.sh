python -m black deepdream_vae
python -m isort deepdream_vae --profile black
python -m mypy --strict deepdream_vae
python -m flake8 deepdream_vae
python -m pip install -e .
python -m pytest deepdream_vae/tests
