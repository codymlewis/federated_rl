from typing import Tuple
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
import numpy as np
import chex
import safeflax


class ReplayMemory:
    def __init__(self, obs_shape: Tuple[int], act_shape: Tuple[int], capacity: int, seed: int = 63):
        self.capacity = capacity
        self.times_pushed = 0
        self.push_index = 0
        self.rng = np.random.default_rng(seed)
        self.observations = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((capacity,) + act_shape, dtype=np.int8)
        self.next_observations = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)

    def push(self, observation: chex.Array, action: chex.Array, next_observation: chex.Array, reward: int):
        self.observations[self.push_index] = observation
        self.actions[self.push_index] = action
        self.next_observations[self.push_index] = next_observation
        self.rewards[self.push_index] = reward
        self.push_index = (self.push_index + 1) % self.capacity
        self.times_pushed += 1

    def sample(self, batch_size: int = 128) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        idx = self.rng.choice(min(self.times_pushed, self.capacity), size=batch_size)
        return self.observations[idx], self.actions[idx], self.next_observations[idx], self.rewards[idx]

    def __len__(self) -> int:
        return min(self.times_pushed, self.capacity)


class DQN(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        return x


@jax.jit
def predict_action(state: train_state.TrainState, observation: chex.Array) -> int:
    return jnp.argmax(state.apply_fn(state.params, observation))


def select_action(
    state: train_state.TrainState,
    observation: chex.Array,
    steps_done: int,
    evaluation: bool = True,
    eps_start: float = 0.9,
    eps_end: float = 0.05,
    eps_decay: int = 1000,
) -> int:
    eps = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    if random.random() > eps or evaluation:
        return predict_action(state, observation)
    return env.action_space.sample()


@jax.jit
def learner_step(
    state: train_state.TrainState,
    target_params: optax.Params,
    obs_batch: chex.Array,
    action_batch: chex.Array,
    next_obs_batch: chex.Array,
    reward_batch: chex.Array,
    gamma: float = 0.99
) -> Tuple[float, train_state.TrainState]:
    next_obs_values = jnp.where(
        jnp.isnan(next_obs_batch).any(1),
        jnp.zeros_like(obs_batch.shape[0]),
        state.apply_fn(target_params, next_obs_batch).max(1),
    )
    expected_state_action_values = (next_obs_values * gamma) + reward_batch

    def loss_fn(params, delta=1.0):
        # Compute Q(s_t, a)
        state_action_values = state.apply_fn(params, obs_batch)[jnp.arange(action_batch.shape[0]), action_batch]
        # Then Huber loss
        dist = jnp.abs(state_action_values - expected_state_action_values)
        return jnp.mean(jnp.where(dist < delta, 0.5 * dist**2 / delta, dist - 0.5 * delta))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -100, 100), grads)
    state = state.apply_gradients(grads=grads)
    return loss, state


class Client:
    def __init__(self, state, target_params, memory):
        self.steps_done = 0
        self.state = state
        self.target_params = target_params
        self.memory = memory


@jax.jit
def fedavg(all_params):
    return jax.tree_util.tree_map(lambda *x: sum(x) / len(x), *all_params)


@jax.jit
def interpolate_params(params_a, params_b, tau):
    return jax.tree_util.tree_map(lambda a, b: b * tau + a * (1 - tau), params_a, params_b)


if __name__ == "__main__":
    batch_size = 128
    tau = 0.005
    num_episodes = 10
    num_clients = 10
    seed = 62
    random.seed(seed)
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=seed)
    model = DQN(env.action_space.n)

    clients = [Client(
        train_state.TrainState.create(
            apply_fn=model.apply,
            params=model.init(jax.random.PRNGKey(seed), observation),
            tx=optax.amsgrad(1e-3),
        ),
        model.init(jax.random.PRNGKey(seed), observation),
        ReplayMemory(observation.shape, (env.action_space.n), 10000)
    ) for _ in range(num_clients)]

    episode_durations = []
    steps_done = 0
    for e in (pbar := trange(num_episodes)):
        episode_durations.append([])
        losses = []
        for c, client in enumerate(clients):
            observation, info = env.reset(seed=round(math.pi * e**2) + seed + c)
            for i in itertools.count():
                observation = jnp.array(observation)
                action = select_action(client.state, observation, client.steps_done, evaluation=False)
                client.steps_done += 1
                new_observation, reward, terminated, truncated, info = env.step(action.item())

                if terminated:
                    next_observation = jnp.full_like(new_observation, jnp.nan)
                else:
                    next_observation = jnp.array(new_observation)

                client.memory.push(observation, action, next_observation, reward)
                observation = next_observation
                if len(client.memory) >= batch_size:
                    batch = client.memory.sample(batch_size)
                    loss, client.state = learner_step(client.state, client.target_params, *batch)
                else:
                    loss = 0.0

                # Soft update of the target network's weights
                client.target_params = interpolate_params(client.target_params, client.state.params, tau)
                if terminated or truncated:
                    losses.append(loss)
                    episode_durations[-1].append(i + 1)
                    break
        global_params = fedavg([client.state.params for client in clients])
        global_target_params = fedavg([client.target_params for client in clients])
        for client in clients:
            client.state = client.state.replace(params=global_params)
            client.target_params = global_target_params
        pbar.set_postfix_str("AVG Loss (STD): {:.5f} ({:.5f}), AVG Dur (STD): {:5.3f} ({:5.3f})".format(
            np.mean(losses), np.std(losses), np.mean(episode_durations[-1]), np.std(episode_durations[-1])
        ))
    env.close()

    print("Average episode durations:")
    print(np.mean(episode_durations, -1))
    save_fn = "trained_model.safetensors"
    safeflax.save_file(global_params, save_fn)
    print(f"Saved the final global model to the {save_fn} file.")
