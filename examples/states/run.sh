XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_score_matching_online.py --config configs/max_entropy_learner_config.py --env_name HalfCheetah-v3 --config.temp 0.2
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_score_matching_online.py --config configs/max_entropy_learner_config.py --env_name Humanoid-v3 --config.temp 0.02
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_score_matching_online.py --config configs/max_entropy_learner_config.py --env_name Ant-v3 --config.temp 0.05
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_score_matching_online.py --config configs/max_entropy_learner_config.py --env_name Walker2d-v3 --config.temp 0.01
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_score_matching_online.py --config configs/max_entropy_learner_config.py --env_name Hopper-v3 --config.temp 0.05
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python3 train_score_matching_online.py --config configs/max_entropy_learner_config.py --env_name Swimmer-v3 --config.temp 0.005