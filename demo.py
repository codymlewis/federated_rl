import itertools
import gymnasium as gym
from flax.training import train_state
import optax
import safeflax

import train


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset()
    model = train.DQN(env.action_space.n)
    params = safeflax.load_file("trained_model.safetensors")
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.identity())

    for i in itertools.count():
        action = train.select_action(state, observation, 0, evaluation=True)
        observation, reward, terminated, truncated, info = env.step(action.item())
        if terminated or truncated:
            break
    print(f"Survived {i + 1} steps")
