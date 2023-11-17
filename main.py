from collections import deque
from typing import NamedTuple
import math
import random
import itertools
import gymnasium as gym
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from tqdm import trange
import chex


class Transition(NamedTuple):
    state: chex.Array
    action: int
    next_state: chex.Array
    reward: float


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        return x


@jax.jit
def select_action(state, observation, key, steps_done, default_action, evaluation: bool):
    eps = 0.05 + (0.9 - 0.05) * jnp.exp(-1. * steps_done / 1000)
    return jnp.where(jax.random.uniform(key) > eps, jnp.argmax(state.apply_fn(state.params, observation)), default_action)


@jax.jit
def learner_step(state, target_params, batch, gamma: float = 0.99):
    state_batch = jnp.array(batch.state)
    action_batch = jnp.array(batch.action)
    reward_batch = jnp.array(batch.reward)
    next_state_batch = jnp.array(batch.next_state)
    next_state_values = jnp.where(
        jnp.isnan(next_state_batch).any(1),
        jnp.zeros_like(state_batch.shape[0]),
        state.apply_fn(target_params, next_state_batch).max(1),
    )
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    def loss_fn(params, delta=1.0):
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = state.apply_fn(params, state_batch)[jnp.arange(action_batch.shape[0]), action_batch]
        # state_action_values = jnp.take_along_axis(
        #     state.apply_fn(params, state_batch),
        #     action_batch.reshape(-1, 1),
        #     axis=1
        # ).reshape(-1)
        dist = jnp.abs(state_action_values - expected_state_action_values)
        return jnp.mean(jnp.where(dist < delta, 0.5 * dist**2 / delta, dist - 0.5 * delta))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -100, 100), grads)
    state = state.apply_gradients(grads=grads)
    return loss, state


if __name__ == "__main__":
    batch_size = 128
    tau = 0.005
    num_episodes = 600
    seed = 62
    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=seed)
    model = DQN(env.action_space.n)
    rng_key = jax.random.PRNGKey(seed)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(rng_key, observation),
        # tx=optax.chain(optax.amsgrad(1e-4), optax.add_decayed_weights(1e-2)),
        tx=optax.amsgrad(1e-3),
    )
    target_params = model.init(rng_key, observation)
    memory = ReplayMemory(10000)
    episode_durations = []
    episode_rng_keys = iter(jax.random.split(rng_key, (num_episodes, 500)))
    steps_done = 0
    for e in (pbar := trange(num_episodes)):
        observation, info = env.reset(seed=round(math.pi * e**2) + seed)
        rng_keys = iter(next(episode_rng_keys))
        for i in itertools.count():
            rng_key = next(rng_keys)
            action = select_action(state, observation, rng_key, steps_done, env.action_space.sample(), evaluation=False)
            steps_done += 1
            new_observation, reward, terminated, truncated, info = env.step(action.item())

            if terminated:
                next_observation = jnp.full_like(new_observation, jnp.nan)
            else:
                next_observation = new_observation

            memory.push(observation, action, next_observation, reward)
            observation = next_observation
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                loss, state = learner_step(state, target_params, batch)
            else:
                loss = 0.0

            # # Soft update of the target network's weights
            # # θ′ ← τ θ + (1 −τ )θ′
            target_params = jax.tree_util.tree_map(lambda tp, op: op * tau + tp * (1 - tau), target_params, state.params)
            if terminated or truncated:
                pbar.set_postfix_str(f"Loss: {loss:.5f}, Reward: {reward:.5f}, Dur: {i + 1}")
                episode_durations.append(i + 1)
                break

    env.close()
    print(episode_durations)
