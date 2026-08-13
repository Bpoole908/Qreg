from typing import Dict
from pdb import set_trace

import torch
from continual_rl.policies.config_base import ConfigBase

from crl.common.yaml import convert_float_to_str as f2s


def get_tag_value(
    dict,
    key,
    template,
    default,
    check_value=lambda x: True,
    transform_value=lambda x: x
):
    """Formats `dict[key]` into `template` if present and valid, else returns a default.

    Args:
        dict: The dict to look up `key` in (e.g. `policy.policy_kwargs`).
        key: The key to look up.
        template: A `str.format`-style template with one placeholder, applied
            to the (possibly transformed) value.
        default: Value to return if `key` is missing or fails `check_value`.
        check_value: Predicate applied to `dict[key]`; if it returns falsy, `default` is used.
        transform_value: Function applied to `dict[key]` before formatting into `template`.

    Returns:
        The formatted string, or `default` if the key is absent/invalid.
    """
    if key in dict and check_value(dict[key]):
        value = transform_value(dict[key])
        return template.format(value)
    else:
        return default


def experiment_tag(policy, task_seq):
    """Builds a short experiment tag string summarizing a DQN policy's non-default hyperparameters.

    Args:
        policy: The policy instance, whose `policy_kwargs` dict is inspected.
       task_seq: The raw task sequence config.

    Returns:
        An underscore-joined tag string (e.g. 'rb=1m_lr=0.0001_rsrb'), including
        only the hyperparameters that differ from `DQNPolicyConfig` defaults or
        are otherwise notable.
    """
    defaults = DQNPolicyConfig()
    pk = policy.policy_kwargs
    tag = [
        get_tag_value(pk, 'buffer_size', 'rb={}', '', transform_value=f2s),
        get_tag_value(pk, 'learning_rate', 'lr={}', ''),
        get_tag_value(pk, 'timesteps_per_collection', 'tpc={}', '', 
                      check_value=lambda x: x if x != defaults.timesteps_per_collection else False),
        get_tag_value(pk, 'update_after', 'upaf={}', '', check_value=lambda x: x, transform_value=f2s),
    ]
    
    # Reset tags
    tag.extend([
        get_tag_value(pk, 'reset_buffer', 'rsrb','', check_value=lambda x: x if x == True else False),
        get_tag_value(pk, 'reset_optimizer', 'rsopt', '', check_value=lambda x: x if x == True else False),
    ])
        
    return '_'.join(filter(None, tag))   


class DQNPolicyConfig(ConfigBase):
    """Hyperparameter/config container for `DQNPolicy`.

    Attributes:
        timesteps_per_collection: Number of timesteps to collect before
            training (i.e. number of timesteps before `train()` is called). By
            default assumes a frameskip wrapper is used, so 4 steps = 16 frames.
        render_collection_freq: Rate at which the agent renders frames as
            videos. Disabled if None.
        reset_buffer: Reset the replay buffer when a task ends.
        reset_optimizer: Reset the optimizer when a task ends.
        update_after: Minimum number of samples needed in the replay buffer
            before training can begin. Defaults to `batch_size` if None, to
            start training as soon as possible.
        buffer_size: Size of the replay buffer. If multiple environments are
            used, `buffer_size // n_envs` where each environment gets a
            proportional segment of the buffer space.
        optimize_memory_usage: If True, the replay buffer stores all
            observations in a single data structure. If False, next
            observations and current observations are stored in two separate
            arrays (roughly 2x the memory required).
        handle_timeout_termination: If True, a terminal state (`done`) caused
            by reaching the max allotted time/frames is not treated as a true
            done. Currently cannot be True when `optimize_memory_usage=True`
            due to a bug. https://github.com/DLR-RM/stable-baselines3/issues/934
        batch_size: Size of the batch when sampling from the replay buffer.
        combined_sampling: If True, the latest sample is always appended to
            the replay sample batch. See https://arxiv.org/abs/1712.01275
        dqn_kwargs: Kwargs for the DQN model, not including observation space
            and output_dims.
        loss_name: Name of the method for computing DQN loss within
            `DQNLosses`. Current options are 'dqn' and 'double_dqn'.
        loss_kwargs: Kwargs for the corresponding loss function given by `loss_name`.
        optimizer_class: Torch optimizer class to use for learning.
        learning_rate: Optimizer learning rate.
        exploration_rate: Epsilon-greedy value for exploration.
        target_update_frequency: Number of timesteps to wait before updating
            the target model.
        save_replay_buffer: If True, saves the replay buffer each time
            `DQNPolicy.save()` is called.
        track_task_id: Whether to store task_id for rehearsal samples. Useful
            for debugging or examining results.
    """
    def __init__(self):
        super().__init__()
        # EnvRunner
        # timesteps_per_collection * num_parallel_envs determines continual_rl step size
        self.timesteps_per_collection: int = 4
        self.render_collection_freq: int = None
        
        # Policy/Model/Optimizer 
        self.dqn_kwargs: dict = {}
        self.loss_name: str = 'dqn'
        self.loss_kwargs: dict = {}
        self.optimizer_class: torch.optim.Optimizer = None # See _set_optimizer_class_defaults()
        self.learning_rate: float = 1e-4
        self.exploration_rate: float = 0.05
        self.target_update_frequency: int = 10000
        self.update_after: int = None
        self.reset_buffer: bool = False
        self.reset_optimizer: bool = False
        self.save_replay_buffer: bool = True
        self.track_task_id: bool = True
        
        # ReplayBuffer kwargs
        self.batch_size: int = 32
        self.buffer_size: int = 1e6
        self.combined_sampling: bool = False
        self.optimize_memory_usage: bool = True
        self.handle_timeout_termination: bool = False

    def _load_from_dict_internal(self, config_dict):
        """Applies overrides from `config_dict`, fills in derived defaults, and coerces numeric fields to int.

        Args:
            config_dict: Dict of config overrides (e.g. from Hydra), applied
                via `ConfigBase._auto_load_class_parameters`.

        Returns:
            self, with fields updated/coerced in place.
        """
        self._auto_load_class_parameters(config_dict)
        self._set_optimizer_class_defaults()
        self.update_after = self.batch_size if self.update_after is None else self.update_after
        
        self.timesteps_per_collection = int(self.timesteps_per_collection)
        self.render_collection_freq = int(self.render_collection_freq) if self.render_collection_freq else None
        self.buffer_size = int(self.buffer_size)
        self.batch_size = int(self.batch_size)
        self.update_after = int(self.update_after)
        self.target_update_frequency = int(self.target_update_frequency)
        return self
    
    def _set_optimizer_class_defaults(self):
        """Defaults `optimizer_class` to `torch.optim.Adam` if it wasn't otherwise set."""
        self.optimizer_class = torch.optim.Adam if self.optimizer_class is None else self.optimizer_class
