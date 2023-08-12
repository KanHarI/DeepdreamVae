from setuptools import setup

__VERSION__ = "0.1.0"

setup(
    name="deepdream_vae",
    version=__VERSION__,
    packages=["deepdream_vae"],
    python_requires=">=3.10",
    install_requires=[
        "dacite",
        "einops",
        "hydra-core",
        "numpy",
        "Pillow",
        "torch",
        "torchaudio",
        "torchvision",
        "tqdm",
        "types-tqdm",
        "wandb",
    ],
    include_package_data=True,
)
