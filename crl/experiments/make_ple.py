from pdb import set_trace

import gym
from crl.common.envs.gym_ple import GymPLE

from continual_rl.utils.env_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    TimeLimit,
    EpisodicLifeEnv,
    FireResetEnv,
)

from continual_rl.experiments.tasks.image_task import ImageTask
from continual_rl.experiments.tasks.make_atari_task import (
    EpisodicLifeEnv,
)

def make_ple(env_id, max_episode_steps=None, env_kwargs=None):
    """Builds a `GymPLE` environment wrapped with standard Atari-style preprocessing.

    Applies a random no-op reset (up to 30 steps) and 4-frame max-and-skip, plus
    an optional episode step limit.

    Args:
        env_id: PLE game name or class, forwarded to `GymPLE`.
        max_episode_steps: If given, wraps the env with a `TimeLimit` of this
            many steps.
        env_kwargs: Optional kwargs forwarded to `GymPLE`.

    Returns:
        The wrapped Gym environment.
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    env = GymPLE(env_id, **env_kwargs)
    # assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    # return only every `skip`-th frame
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    
    return env

def wrap_ple(env, episode_life=True):
    """Applies additional Atari-style wrappers on top of a (already-`make_ple`-wrapped) PLE env.

    Also wraps with `FireResetEnv` if the game's action set includes a 'FIRE'
    action (some games require pressing fire to start).

    Args:
        env: The environment to wrap.
        episode_life: If True, wraps with `EpisodicLifeEnv` so losing a life
            ends the episode (evaluation only resets on true game over).

    Returns:
        The wrapped environment.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    return env

def get_single_ple_task(
    task_id,
    action_space_id,
    env_name,
    num_timesteps,
    continual_eval_num_returns=10,
    max_episode_steps=None,
    env_kwargs=None
):
    """Builds a `continual_rl` `ImageTask` for a single PLE game.

    The env is constructed lazily via a closure (`env_spec`) so that each
    task's `env_name` is captured correctly rather than shared/mutated across
    tasks built in a loop (see `exp_loader`). Images are preprocessed to 84x84
    grayscale with a 4-frame time batch, matching the typical Atari default.

    Args:
        task_id: Identifier for this task.
        action_space_id: Identifier for this task's action space (tasks sharing
            an action space id share network output heads).
        env_name: PLE game name/class to build (forwarded to `make_ple`).
        num_timesteps: Number of training timesteps for this task.
        continual_eval_num_returns: Number of episode returns to average over
            during continual evaluation.
        max_episode_steps: If given, caps episode length (forwarded to
            `make_ple`); the Atari-standard default there is 100k steps.
        env_kwargs: Optional kwargs forwarded to `make_ple`/`GymPLE`.

    Returns:
        A `continual_rl.experiments.tasks.image_task.ImageTask` for this game.
    """
    env_spec = lambda: wrap_ple(
            make_ple(env_name, max_episode_steps=max_episode_steps, env_kwargs=env_kwargs),
        )

    image_task = ImageTask(
        task_id=task_id,
        action_space_id=action_space_id,
        env_spec=env_spec,
        num_timesteps=num_timesteps,
        time_batch_size=4,
        eval_mode=False,
        image_size=[84, 84],
        grayscale=True,
        continual_eval_num_returns=continual_eval_num_returns,
    )
    
    return image_task