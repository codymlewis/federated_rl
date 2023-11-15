from collections import deque, namedtuple
import math
import random
import itertools
import gymnasium as gym
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import rlax
from tqdm import trange


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


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


def select_action(state, observation, key, evaluation: bool):
    q = state.apply_fn(state.params, observation)
    eps = 0.05 + (0.9 - 0.05) * math.exp(-1. * state.step / 1000)
    train_a = rlax.epsilon_greedy(eps).sample(key, q)
    eval_a = rlax.greedy().sample(key, q)
    a = jax.lax.select(evaluation, eval_a, train_a)
    return a


def learner_step(state, target_params, memory, batch_size: int = 128):
    if len(memory) < batch_size:
        return 0.0, state
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    non_final_mask = jnp.array(tuple(map(lambda s: s is not None, batch.next_state)), dtype=bool)
    non_final_next_states = jnp.array([s for s in batch.next_state if s is not None])

    state_batch = jnp.array(batch.state)
    action_batch = jnp.array(batch.action)
    reward_batch = jnp.array(batch.reward)
    next_state_values = jnp.zeros(batch_size)
    next_state_values.at[non_final_mask].set(state.apply_fn(target_params, non_final_next_states).max(1))
    expected_state_action_values = (next_state_values * 0.99) + reward_batch

    def loss_fn(params, delta=1.0):
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = jnp.take(state.apply_fn(state.params, state_batch), action_batch, axis=1)
        expected_sav = jax.nn.one_hot(expected_state_action_values, state_action_values.shape[-1])
        dist = jnp.abs(state_action_values - expected_sav)
        return jnp.mean(jnp.where(dist < delta, 0.5 * dist**2, delta * (dist - 0.5 * delta)))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state


if __name__ == "__main__":
    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    model = DQN(env.action_space.n)
    rng_key = jax.random.PRNGKey(42)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(rng_key, observation),
        tx=optax.adamw(1e-4),
    )
    target_params = state.params.copy()
    memory = ReplayMemory(10000)

    episode_durations = [0]
    for i in (pbar := trange(500)):
        observation, info = env.reset()
        for i in itertools.count():
            rng_key = jax.random.split(rng_key, 1)[0]
            action = select_action(state, observation, rng_key, evaluation=False)
            observation, reward, terminated, truncated, info = env.step(action.item())
            episode_durations[-1] += 1

            next_observation = observation

            memory.push(observation, action, next_observation, reward)
            observation = next_observation
            loss, state = learner_step(state, target_params, memory)
            pbar.set_postfix_str(f"Loss: {loss:.5f}")

            # # Soft update of the target network's weights
            # # θ′ ← τ θ + (1 −τ )θ′
            tau = 0.005
            target_params = jax.tree_util.tree_map(lambda tp, op: op * tau + tp * (1 - tau), target_params, state.params)
            if terminated or truncated:
                episode_durations.append(0)
                break

    env.close()
    print(episode_durations)
