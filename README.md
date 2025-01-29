# Maximum Entropy RL with Diffusion Policy (MaxEntDP)

## Overview

This repository provides the implementation of the MaxEntDP algorithm for the paper "Maximum Entropy Reinforcement Learning with Diffusion Policy".

## Installation

To get started, you need to install the required dependencies.

```bash
conda create -n MaxEntDP python=3.9
conda activate MaxEntDP
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Getting Started

To reproduce the results in the paper, navigate to the respective example directories and execute the provided training script:

```bash
cd examples/states
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_score_matching_online.py --config configs/max_entropy_learner_config.py --env_name HalfCheetah-v3 --config.temp 0.2
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_score_matching_online.py --config configs/max_entropy_learner_config.py --env_name Humanoid-v3 --config.temp 0.02
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_score_matching_online.py --config configs/max_entropy_learner_config.py --env_name Ant-v3 --config.temp 0.05
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_score_matching_online.py --config configs/max_entropy_learner_config.py --env_name Walker2d-v3 --config.temp 0.01
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_score_matching_online.py --config configs/max_entropy_learner_config.py --env_name Hopper-v3 --config.temp 0.02
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_score_matching_online.py --config configs/max_entropy_learner_config.py --env_name Swimmer-v3 --config.temp 0.005
```
When running with multiple gpus, the batch size (default 256) should be divisible by the number of devices.

## Important Files and Scripts

- **[Main Training Script](examples/states/train_score_matching_online.py)**: The main training script to train a diffusion model agent using MaxEntDP. Includes options for the environment and training scenario.

- **[MaxEntDP Learner](jaxrl5/agents/score_matching/max_entropy_learner.py)**: The core implementation of the MaxEntDP algorithm, including methods for creating the learner, updating critic and actor networks, and sampling actions. **Note** that if you want to make any changes to the learner after installation, you will need to reinstall jaxrl5 locally, by running the following from the root directory of the repository:
```bash
pip install ./
```

- **[Training Configuration for MaxEntDP](examples/states/configs/max_entropy_learner_config.py)**: Configuration file for setting hyperparameters and model configurations for the MaxEntDP learner.

- **[DDPM Implementation](jaxrl5/networks/diffusion.py)**: Contains the implementation of Denoising Diffusion Probabilistic Models (DDPM).

The code is built on top of the [QSM](https://github.com/Alescontrela/score_matching_rl) implementation.
