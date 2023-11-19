# Federated Reinforcement Learning

A demonstration of deep reinforcement learning within a federated learning setting.

## Setup

You will need to install [Python 3.10+](python.org) and [poetry](python-poetry.org), then run the `configure.sh` bash script
in the root of this repository.

You may need to install the [swig](https://swig.org/) library.

## Run the Code

After setting up, you can perform training with `poetry run python train.py`, this will take a while.

After training, you can demo the resulting model with `poetry run python demo.py`.

We provide a pretrained model in the folder `trained_model`, you can use this to skip training.
