import itertools
import jax
import gymnasium as gym
from flax.training import train_state, orbax_utils
import orbax.checkpoint as ocp
import optax

import main


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset()
    ckpt_mgr = ocp.CheckpointManager(
        "trained_model", ocp.Checkpointer(ocp.PyTreeCheckpointHandler()), options=None
    )
    model = main.DQN(env.action_space.n)
    params = model.init(jax.random.PRNGKey(0), observation)
    params = ckpt_mgr.restore(
        ckpt_mgr.latest_step(),
        params,
        restore_kwargs={'restore_args': orbax_utils.restore_args_from_target(params, mesh=None)}
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.identity())

    for i in itertools.count():
        action = main.select_action(state, observation, 0, evaluation=True)
        observation, reward, terminated, truncated, info = env.step(action.item())
        if terminated or truncated:
            break
    print(f"Survived {i + 1} steps")
