from functools import partial
from typing import Callable, Optional, Sequence, Type
import flax.linen as nn
import jax.numpy as jnp
import jax

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    betas = jnp.linspace(
        beta_start, beta_end, timesteps
    )
    return betas

def vp_beta_schedule(timesteps):
    t = jnp.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas

class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)

class DDPM(nn.Module):
    cond_encoder_cls: Type[nn.Module]
    reverse_encoder_cls: Type[nn.Module]
    time_preprocess_cls: Type[nn.Module]

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray,
                 a: jnp.ndarray,
                 time: jnp.ndarray,
                 training: bool = False):

        t_ff = self.time_preprocess_cls()(time)
        cond = self.cond_encoder_cls()(t_ff, training=training)
        reverse_input = jnp.concatenate([a, s, cond], axis=-1)

        return self.reverse_encoder_cls()(reverse_input, training=training)

@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'clip_sampler', 'training'))
def ddpm_sampler(actor_apply_fn, actor_params, T, rng, act_dim, observations, clip_sampler, training = False):

    batch_size = observations.shape[0]
    noise_schedule = NoiseScheduleVP(schedule='cosine')
    steps = jnp.linspace(1e-3, noise_schedule.T, T+1)
    alpha_hat_prevs = noise_schedule.marginal_alpha(steps) ** 2
    alpha_hats = alpha_hat_prevs[1:]
    alphas = alpha_hat_prevs[1:] / alpha_hat_prevs[:-1]
    betas = 1 - alphas
    
    def fn(input_tuple, time):
        current_x, rng = input_tuple
        
        input_time = jnp.expand_dims(
            jnp.array([time]).repeat(current_x.shape[0]), axis=1)
        log_snr = jnp.log(alpha_hats[input_time]) - jnp.log(1 - alpha_hats[input_time])
        eps_pred = actor_apply_fn(
            {"params": actor_params},
            observations, current_x,
            log_snr, training=training)
        
        x_start = (current_x - jnp.sqrt(1 - alpha_hats[time]) * eps_pred) / jnp.sqrt(alpha_hats[time])
        x_start = jnp.clip(x_start, -1, 1) if clip_sampler else x_start

        mean_coef1 = jnp.sqrt(alpha_hat_prevs[time]) * betas[time] / (1 - alpha_hats[time])
        mean_coef2 = jnp.sqrt(alphas[time]) * (1 - alpha_hat_prevs[time]) / (1 - alpha_hats[time])
        mean = mean_coef1 * x_start + mean_coef2 * current_x

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(
            key, shape=(observations.shape[0], current_x.shape[1]),)

        current_x = mean + (time > 0) * (jnp.sqrt(betas[time]) * z)
        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    (action_0, rng), () = jax.lax.scan(
        fn, (jax.random.normal(key, (batch_size, act_dim)), rng),
        jnp.arange(T-1, -1, -1), unroll=5)
    action_0 = jnp.clip(action_0, -1, 1)
    return action_0, rng

@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'clip_sampler', 'training'))
def ddpm_sampler_qsm(actor_apply_fn, actor_params, T, rng, act_dim, observations, alphas, alpha_hats, betas, sample_temperature, clip_sampler, training = False):

    batch_size = observations.shape[0]
    
    def fn(input_tuple, time):
        current_x, rng = input_tuple
        
        input_time = jnp.expand_dims(
            jnp.array([time]).repeat(current_x.shape[0]), axis=1)
        eps_pred = actor_apply_fn(
            {"params": actor_params},
            observations, current_x,
            input_time, training=training)

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(
            key, shape=(observations.shape[0], current_x.shape[1]),)

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)
        current_x = jnp.clip(current_x, -1, 1) if clip_sampler else current_x
        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    (action_0, rng), () = jax.lax.scan(
        fn, (jax.random.normal(key, (batch_size, act_dim)), rng),
        jnp.arange(T-1, -1, -1), unroll=5)
    action_0 = jnp.clip(action_0, -1, 1)
    return action_0, rng

from jaxrl5.networks.dpm_solver_jax import NoiseScheduleVP, model_wrapper, DPM_Solver

@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'training'))
def dpm_solver_sampler(actor_apply_fn, actor_params, T, rng, act_dim, observations, training = False):

    noise_schedule = NoiseScheduleVP(schedule='cosine')
    def model(action_t, t):
        t = jnp.expand_dims(t, axis=1)
        alpha_hats = noise_schedule.marginal_alpha(t) ** 2
        log_snr = jnp.log(alpha_hats) - jnp.log(1 - alpha_hats)
        return actor_apply_fn(
            {"params": actor_params},
            observations, action_t,
            log_snr, training=training)
    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="noise",  # or "x_start" or "v" or "score"
    )
    dpm_solver = DPM_Solver(model_fn, noise_schedule)

    key, rng = jax.random.split(rng, 2)
    batch_size = observations.shape[0]
    x_T = jax.random.normal(key, (batch_size, act_dim))
    action_0 = dpm_solver.sample(
        x_T,
        steps=T,
        order=3,
        skip_type="time_uniform",
        method="singlestep",
    )

    action_0 = jnp.clip(action_0, -1, 1)
    return action_0, rng

@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'clip_sampler', 'training'))
def ddpm_sampler_keepinner(actor_apply_fn, actor_params, T, rng, act_dim, observations, alphas, alpha_hats, betas, sample_temperature, clip_sampler, training = False):

    batch_size = observations.shape[0]
    
    def fn(input_tuple, time):
        current_x, logprob_total, rng = input_tuple

        input_time = jnp.expand_dims(
            jnp.array([time]).repeat(current_x.shape[0]), axis=1)
        eps_pred = actor_apply_fn(
            {"params": actor_params},
            observations, current_x,
            input_time, training=training)

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x_plus = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(
            key, shape=(observations.shape[0], current_x.shape[1]),)
        z_scaled = sample_temperature * z
        current_z = current_x_plus + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        # compute logprob of current_z, scaled norm with respect to current_x
        logprobs = -0.5 * jnp.sum((current_z - current_x) ** 2, axis=-1)
        # scale and normalize if nonzero variance, otherwise set to 0
        condition = (time > 0) & (jnp.sqrt(betas[time]) * sample_temperature > 0)
        logprobs = jax.lax.cond(condition, 
             lambda _: (1 / (betas[time] * sample_temperature**2))* logprobs, 
             lambda _: jnp.zeros_like(logprobs),
             None)
        logprob_total = logprob_total + logprobs

        current_x = jnp.clip(current_z, -1, 1) if clip_sampler else current_x

        return (current_x, logprob_total, rng), ()

    key, rng = jax.random.split(rng, 2)
    (action_0, logprob_total, rng), () = jax.lax.scan(
        fn, (jax.random.normal(key, (batch_size, act_dim)), jnp.zeros(batch_size), rng),
        jnp.arange(T-1, -1, -1), unroll=5)
    action_0 = jnp.clip(action_0, -1, 1)
    logprob_total = (1/T)*logprob_total

    # new actions have the logprob totals appended at the end of each action
    # log_prob_total = jnp.expand_dims(logprob_total, axis=-1)
    # all_actions = jnp.concatenate((action_0, log_prob_total), axis=1)
    # assert all_actions.shape == (batch_size, act_dim+1)
    return action_0, logprob_total, rng