from collections import deque
from typing import Tuple, List, NamedTuple
import math
import random
import itertools
import gymnasium as gym
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, orbax_utils
import optax
from tqdm import trange
import numpy as np
import chex
import orbax.checkpoint as ocp


class Transition(NamedTuple):
    state: chex.Array
    action: int
    next_state: chex.Array
    reward: float


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


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
    state: train_state.TrainState, target_params: optax.Params, batch: Transition, gamma: float = 0.99
) -> Tuple[float, train_state.TrainState]:
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
        # Compute Q(s_t, a)
        state_action_values = state.apply_fn(params, state_batch)[jnp.arange(action_batch.shape[0]), action_batch]
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


if __name__ == "__main__":
    batch_size = 128
    tau = 0.005
    num_episodes = 400
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
        ReplayMemory(10000)
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
                    transitions = client.memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))
                    loss, client.state = learner_step(client.state, client.target_params, batch)
                else:
                    loss = 0.0

                # Soft update of the target network's weights
                client.target_params = jax.tree_util.tree_map(
                    lambda tp, op: op * tau + tp * (1 - tau), client.target_params, client.state.params)
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
    print()

    ckpt_mgr = ocp.CheckpointManager(
        "trained_model", ocp.Checkpointer(ocp.PyTreeCheckpointHandler()), options=ocp.CheckpointManagerOptions(create=True)
    )
    ckpt_mgr.save(num_episodes, global_params, save_kwargs={'save_args': orbax_utils.save_args_from_target(global_params)})
    print("Saved the final global model to the 'trained_model' folder.")
