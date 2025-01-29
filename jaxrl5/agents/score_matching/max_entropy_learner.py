"""Implementations of algorithms for continuous control."""
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax import struct

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, StateActionValue, DDPM, FourierFeatures
from jaxrl5.networks import cosine_beta_schedule, vp_beta_schedule, ddpm_sampler, dpm_solver_sampler
from jaxrl5.networks.dpm_solver_jax import NoiseScheduleVP

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


def tensorstats(tensor, prefix=None):
  assert tensor.size > 0, tensor.shape
  metrics = {
      'mean': tensor.mean(),
      'std': tensor.std(),
      'mag': jnp.abs(tensor).max(),
      'min': tensor.min(),
      'max': tensor.max(),
  }
  if prefix:
    metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
  return metrics


class MaxEntropyLearner(Agent):
    score_model: TrainState
    critic_1: TrainState
    critic_2: TrainState
    target_critic_1: TrainState
    target_critic_2: TrainState

    discount: float
    tau: float
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    clip_sampler: bool = struct.field(pytree_node=False)
    temp: float
    backup_entropy: bool = struct.field(pytree_node=False)
    samples_num: int = struct.field(pytree_node=False)
    T_log_prob: int = struct.field(pytree_node=False)
    eval_action_selection : bool
    eval_candidate_num: int = struct.field(pytree_node=False)
    score_samples_num: int = struct.field(pytree_node=False)
    policy_training: str = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_architecture: str = 'mlp',
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        actor_layer_norm: bool = False,
        T: int = 20,
        time_dim: int = 64,
        clip_sampler: bool = True,
        decay_steps: Optional[int] = int(2e6),
        temp: float = 0.1,
        backup_entropy: bool = True,
        samples_num: int = 50,
        T_log_prob: int = 20,
        eval_action_selection : bool = True,
        eval_candidate_num: int = 10,
        score_samples_num: int = 500,
        policy_training: str = 'QNE',
    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[-1]

        # Time embedding network.
        preprocess_time_cls = partial(
            FourierFeatures, output_size=time_dim, learnable=True)

        cond_model_cls = partial(
            MLP, hidden_dims=(128, 128), activations=mish,
            activate_final=False)
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if actor_architecture == 'mlp':
            base_model_cls = partial(MLP,
                hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]),
                activations=mish, use_layer_norm=actor_layer_norm,
                activate_final=False)
            
            actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                             cond_encoder_cls=cond_model_cls,
                             reverse_encoder_cls=base_model_cls)
        else:
            raise ValueError(f'Invalid actor architecture: {actor_architecture}')
        
        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        actor_params = actor_def.init(
            actor_key, observations, actions, time)['params']
        score_model = TrainState.create(
            apply_fn=actor_def.apply, params=actor_params,
            tx=optax.adam(learning_rate=actor_lr))

        # Initialize critics.
        critic_base_cls = partial(
            MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        critic_def = StateActionValue(critic_base_cls)
        critic_key_1, critic_key_2 = jax.random.split(critic_key, 2)
        critic_params_1 = critic_def.init(critic_key_1, observations, actions)["params"]
        critic_params_2 = critic_def.init(critic_key_2, observations, actions)["params"]
        critic_1 = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params_1,
            tx=optax.adam(learning_rate=critic_lr))
        critic_2 = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params_2,
            tx=optax.adam(learning_rate=critic_lr))

        target_critic_def = StateActionValue(critic_base_cls)
        target_critic_1 = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params_1,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),)
        target_critic_2 = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params_2,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),)

        return cls(
            actor=None,
            score_model=score_model,
            critic_1=critic_1,
            critic_2=critic_2,
            target_critic_1=target_critic_1,
            target_critic_2=target_critic_2,
            tau=tau,
            discount=discount,
            rng=rng,
            act_dim=action_dim,
            T=T,
            clip_sampler=clip_sampler,
            temp=temp,
            backup_entropy=backup_entropy,
            samples_num=samples_num,
            T_log_prob=T_log_prob,
            eval_action_selection=eval_action_selection,
            eval_candidate_num=eval_candidate_num,
            score_samples_num=score_samples_num,
            policy_training=policy_training,
        )

    def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        (B, _) = batch['observations'].shape
        (_, A) = batch['actions'].shape

        # Sample actions for next state.
        key, rng = jax.random.split(agent.rng)
        next_actions, rng = ddpm_sampler(
            agent.score_model.apply_fn,
            agent.score_model.params,
            agent.T, rng, agent.act_dim,
            batch['next_observations'],
            agent.clip_sampler)
        assert next_actions.shape == (B, A)

        # Compute target q.
        next_q_1 = agent.target_critic_1.apply_fn(
            {"params": agent.target_critic_1.params}, batch["next_observations"], next_actions)
        next_q_2 = agent.target_critic_2.apply_fn(
            {"params": agent.target_critic_2.params}, batch["next_observations"], next_actions)
        next_v = jnp.stack([next_q_1, next_q_2], 0).min(0)

        # compute log probality
        if agent.backup_entropy:
            log_prob, agent, _, _, _ = agent.calc_log_prob(batch["next_observations"], next_actions)
            next_v = next_v - agent.temp * log_prob

        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v
        metrics = tensorstats(target_q, 'target_q')
        if agent.backup_entropy:
            metrics.update({'log_prob': log_prob.mean()})
        metrics.update({'next_v': next_v.mean()})
        key, rng = jax.random.split(rng)
        assert target_q.shape == (B,)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            q = agent.critic_1.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"])
            loss = ((q - sg(target_q)) ** 2)
            assert loss.shape == (B,)
            metrics = {**tensorstats(loss, 'c_loss'), **tensorstats(q, 'q')}
            return loss.mean(), metrics

        grads_c_1, metrics_c_1 = jax.grad(critic_loss_fn, has_aux=True)(agent.critic_1.params)
        metrics.update({f'{k}_1': v for k, v in metrics_c_1.items()})
        critic_1 = agent.critic_1.apply_gradients(grads=grads_c_1)

        grads_c_2, metrics_c_2 = jax.grad(critic_loss_fn, has_aux=True)(agent.critic_2.params)
        metrics.update({f'{k}_2': v for k, v in metrics_c_2.items()})
        critic_2 = agent.critic_2.apply_gradients(grads=grads_c_2)

        target_critic_1_params = optax.incremental_update(
            critic_1.params, agent.target_critic_1.params, agent.tau)
        target_critic_2_params = optax.incremental_update(
            critic_2.params, agent.target_critic_2.params, agent.tau)
        target_critic_1 = agent.target_critic_1.replace(params=target_critic_1_params)
        target_critic_2 = agent.target_critic_2.replace(params=target_critic_2_params)
        new_agent = agent.replace(
            critic_1=critic_1, critic_2=critic_2,
            target_critic_1=target_critic_1,
            target_critic_2=target_critic_2,
            rng=rng)
        return new_agent, metrics

    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        B, A = batch['actions'].shape
        noise_schedule = NoiseScheduleVP(schedule='cosine')

        # Forward process with RB actions.
        key, rng = jax.random.split(agent.rng, 2)
        time = jax.random.uniform(key, (B,), minval=1e-3, maxval=noise_schedule.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(
            key, (B, agent.act_dim))
        alpha_hats = noise_schedule.marginal_alpha(time) ** 2
        log_snr = jnp.log(alpha_hats) - jnp.log(1 - alpha_hats)
        log_snr = jnp.expand_dims(log_snr, axis=1)

        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample
        key, rng = jax.random.split(rng, 2)

        K = agent.score_samples_num
        noisy_actions_repeat = jnp.repeat(jnp.expand_dims(noisy_actions, axis=1), axis=1, repeats=K) # [B, K, A]
        std = jnp.expand_dims(alpha_2 / alpha_1, axis=-1)
        lower_bound = -1 / alpha_2[:, :, None] * noisy_actions_repeat - (1 / std)
        upper_bound = -1 / alpha_2[:, :, None] * noisy_actions_repeat + (1 / std)
        tnormal_noise = jax.random.truncated_normal(
            key, lower=lower_bound, upper=upper_bound, shape=(B, K, agent.act_dim))
        key, rng = jax.random.split(rng, 2)
        normal_noise = jax.random.normal(key, shape=((B, K, agent.act_dim)))
        normal_noise_clip = jnp.clip(normal_noise, min=lower_bound, max=upper_bound)
        # jax.random.truncated_normal() generates NaN occasionally, so use clipped normal noise to replace NaN
        noise = jnp.where(jnp.isnan(tnormal_noise), normal_noise_clip, tnormal_noise)
        clean_samples = 1 / alpha_1[:, :, None] * noisy_actions_repeat + std * noise
        key, rng = jax.random.split(rng, 2)
        observations_repeat = jnp.repeat(jnp.expand_dims(batch['observations'], axis=1), axis=1, repeats=K)

        devices = jax.devices()
        assert B % len(devices) == 0

        # Compute Q
        # @partial(shard_map, mesh=Mesh(devices, ('i',)), in_specs=(P('i'), P('i')), out_specs=(P('i')))
        def compute_Q(actions, observations):
            critic_1 = agent.target_critic_1.apply_fn(
                {"params": agent.target_critic_1.params}, observations, actions)
            critic_2 = agent.target_critic_2.apply_fn(
                {"params": agent.target_critic_2.params}, observations, actions)
            critic = jnp.stack([critic_1, critic_2], 0).min(0)
            return critic
        
        if agent.policy_training == 'QNE':  # Q-weighted Noise Estimation
            compute_Q_DDP = partial(shard_map, mesh=Mesh(devices, ('i',)), in_specs=(P('i'), P('i')), out_specs=(P('i')))(compute_Q)
            critic = compute_Q_DDP(clean_samples, observations_repeat)
            weight = nn.softmax((1 / agent.temp) * critic, axis=1)
            eps_estimation = -jnp.sum(weight[:, :, None] * noise, axis=1)

        elif agent.policy_training == 'iDEM':  # iterated Denoising Energy Matching
            value_and_grad_vmap = jax.vmap(jax.value_and_grad(compute_Q), in_axes=(0, 0))
            value_and_grad_DDP = partial(shard_map, mesh=Mesh(devices, ('i',)), in_specs=(P('i'), P('i')), out_specs=(P('i')))(value_and_grad_vmap)
            critic, critic_grads = value_and_grad_DDP(
                jnp.reshape(clean_samples, shape=(B * K, A)), jnp.reshape(observations_repeat, shape=(B * K, -1))
            )
            critic = jnp.reshape(critic, shape=(B, K))
            critic_grads = jnp.reshape(critic_grads, shape=(B, K, A))
            critic_grads = (1 / agent.temp) * critic_grads
            weight = nn.softmax((1 / agent.temp) * critic, axis=1)
            eps_estimation = -(alpha_2 / alpha_1) * jnp.sum(weight[:, :, None] * critic_grads, axis=1)
        
        elif agent.policy_training == 'QSM':  # Q Score Matching
            grad_vmap = jax.vmap(jax.grad(compute_Q), in_axes=(0, 0))
            grad_DDP = partial(shard_map, mesh=Mesh(devices, ('i',)), in_specs=(P('i'), P('i')), out_specs=(P('i')))(grad_vmap)
            critic_grads = grad_DDP(noisy_actions, batch['observations'])
            eps_estimation = -alpha_2 * (1 / agent.temp) * critic_grads

        else:
            raise ValueError(f'Unsupported training method: {agent.policy_training}')

        def actor_loss_fn(
                score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = agent.score_model.apply_fn(
                {'params': score_model_params}, batch['observations'], noisy_actions, log_snr)
            assert eps_pred.shape == (B, A)
            actor_loss = jnp.power(sg(eps_estimation) - eps_pred, 2).mean(-1)
            assert actor_loss.shape == (B,)
            metrics = tensorstats(actor_loss, 'actor_loss')
            metrics.update(tensorstats(eps_pred, 'eps_pred'))
            metrics.update(tensorstats(eps_estimation, 'eps_estimation'))
            return actor_loss.mean(0), metrics

        key, rng = jax.random.split(rng, 2)
        grads, metrics = jax.grad(actor_loss_fn, has_aux=True)(
            agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)
        new_agent = agent.replace(
            score_model=score_model,
            rng=rng)
        return new_agent, metrics

    @jax.jit
    def sample_actions(self, observations: jnp.ndarray):
        return self.eval_actions_sample(observations)
    
    def eval_actions(self, observations: jnp.ndarray):
        if self.eval_action_selection:
            return self.eval_actions_select(observations, self.eval_candidate_num)
        else:
            return self.eval_actions_sample(observations)

    @jax.jit
    def eval_actions_sample(self, observations: jnp.ndarray):
        rng = self.rng
        assert len(observations.shape) == 1
        observations = observations[None]

        actions, rng = ddpm_sampler(
            self.score_model.apply_fn,
            self.score_model.params,
            self.T, rng, self.act_dim, observations,
            self.clip_sampler)
        assert actions.shape == (1, self.act_dim)
        _, rng = jax.random.split(rng, 2)
        return jnp.squeeze(actions), self.replace(rng=rng)
    
    @partial(jax.jit, static_argnames='cand_num')
    def eval_actions_select(self, observations: jnp.ndarray, cand_num: int = 10):
        rng = self.rng
        assert len(observations.shape) == 1
        observations = observations[None]
        observations = jnp.repeat(observations, repeats=cand_num, axis=0)

        actions, rng = ddpm_sampler(
            self.score_model.apply_fn,
            self.score_model.params,
            self.T, rng, self.act_dim, observations,
            self.clip_sampler)
        
        q_1 = self.target_critic_1.apply_fn(
            {"params": self.target_critic_1.params}, observations, actions)
        q_2 = self.target_critic_2.apply_fn(
            {"params": self.target_critic_2.params}, observations, actions)
        Q = jnp.stack([q_1, q_2], 0).min(0)

        actions = actions[jnp.argmax(Q, axis=0)]
        assert actions.shape == (self.act_dim,)
        _, rng = jax.random.split(rng, 2)
        return actions, self.replace(rng=rng)
    
    @jax.jit
    def eval_actions_sample_batch(self, observations: jnp.ndarray):
        rng = self.rng

        actions, rng = ddpm_sampler(
            self.score_model.apply_fn,
            self.score_model.params,
            self.T, rng, self.act_dim, observations,
            self.clip_sampler)
        _, rng = jax.random.split(rng, 2)
        return actions, self.replace(rng=rng)
    
    @jax.jit 
    def calc_log_prob(agent, observations, actions):
        samples_num = agent.samples_num
        B, A = actions.shape
        rng = agent.rng
        T = agent.T_log_prob

        time = jnp.arange(0, T)
        noise_schedule = NoiseScheduleVP(schedule='cosine')
        steps = jnp.linspace(1e-3, noise_schedule.T, T+1)
        alpha_hat_prevs = noise_schedule.marginal_alpha(steps) ** 2
        alpha_hats = alpha_hat_prevs[1:]

        time = jnp.expand_dims(time, axis = (0, 1, 3))
        time = jnp.tile(time, reps=(B, samples_num, 1, 1)) # [B, N, T, 1]
        key, rng = jax.random.split(rng)
        noise_sample = jax.random.normal(
            key, (B, samples_num, T, A)) # [B, N, T, A]
        key, rng = jax.random.split(rng, 2)
        alpha_hats = alpha_hats[time]
        alpha_hat_prevs = alpha_hat_prevs[time]
        log_snr = jnp.log(alpha_hats) - jnp.log(1 - alpha_hats)
        alpha_1 = jnp.sqrt(alpha_hats)
        alpha_2 = jnp.sqrt(1 - alpha_hats)
        actions_repeat = jnp.expand_dims(actions, axis=(1, 2))
        actions_repeat = jnp.tile(actions_repeat, reps=(1, samples_num, T, 1)) # [B, N, T, A]
        observations_repeat = jnp.expand_dims(observations, axis=(1, 2))
        observations_repeat = jnp.tile(observations_repeat, reps=(1, samples_num, T, 1)) # [B, N, T, S]
        noisy_actions = alpha_1 * actions_repeat + alpha_2 * noise_sample

        devices = jax.devices()
        assert B % len(devices) == 0

        @partial(shard_map, mesh=Mesh(devices, ('i',)), in_specs=(P('i'), P('i'), P('i')), out_specs=P('i'))
        def predict_eps(obss, noisy_actions, log_snr):

            eps_pred = agent.score_model.apply_fn(
                {'params': agent.score_model.params}, obss,
                noisy_actions, log_snr, training=False)
            return eps_pred

        eps_pred = predict_eps(observations_repeat, noisy_actions, log_snr)
        x_start = (noisy_actions - jnp.sqrt(1 - alpha_hats) * eps_pred) / jnp.sqrt(alpha_hats)
        x_start = jnp.clip(x_start, -1, 1) if agent.clip_sampler else x_start
        eps_pred = (noisy_actions - jnp.sqrt(alpha_hats) * x_start) / jnp.sqrt(1 - alpha_hats)

        key, rng = jax.random.split(rng, 2)
        weight = ((alpha_hat_prevs - alpha_hats) / (2 * alpha_hats * (1 - alpha_hats)))[:, 0, :, 0]
        alphahat_minus_error = (alpha_hats[:, 0, :, 0] * A - ((eps_pred - noise_sample) ** 2).sum(axis=-1).mean(axis=1)) * weight
        log_prob = alphahat_minus_error.sum(axis=1) - 0.5 * A * jnp.log(2 * jnp.pi * jnp.exp(1))

        integral1 = alpha_hats[:, 0, :, 0] * A
        integral2 = ((eps_pred - noise_sample) ** 2).sum(axis=-1).mean(axis=1)
        
        return log_prob, agent.replace(rng=rng), integral1, integral2, log_snr[:, 0, :, 0]
    
    def calc_value(self, observations, actions):
        q_1 = self.target_critic_1.apply_fn(
            {"params": self.target_critic_1.params}, observations, actions)
        q_2 = self.target_critic_2.apply_fn(
            {"params": self.target_critic_2.params}, observations, actions)
        Q = jnp.stack([q_1, q_2], 0).min(0)

        return Q

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        new_agent, critic_info = new_agent.update_q(batch)
        new_agent, actor_info = new_agent.update_actor(batch)
        return new_agent, {**actor_info, **critic_info}
