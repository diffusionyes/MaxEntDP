import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.model_cls = "MaxEntropyLearner"
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.discount = 0.99
    config.tau = 0.005  # For soft target updates.
    config.T = 20  # diffusion timesteps
    config.critic_hidden_dims=(256,256)
    config.actor_hidden_dims=(256,256)
    config.temp = 0.1  # temperature coefficient
    config.backup_entropy = True  # backup entropy when computing Q
    config.samples_num = 50  # For log probability computation
    config.T_log_prob = 20  # For log probability computation
    config.eval_action_selection = True  # use action section when testing
    config.eval_candidate_num = 10  # the number of action candidates
    config.score_samples_num = 500  # For target score estimation
    return config
