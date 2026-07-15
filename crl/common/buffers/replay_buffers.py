"""
The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, make_dataclass
from typing import Any, Dict, Tuple, Union
from pdb import set_trace

import numpy as np
import torch
from gym import spaces

from crl.common.utils import reduce_dims, get_obs_shape, get_action_dim, get_device

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


@dataclass(repr=False)
class Samples():
    """A batch of replay-buffer transitions, functioning like a mutable namedtuple.

    Stores one field per transition component (observations, actions, etc.),
    all sharing a single storage type (either `np.ndarray` or `torch.Tensor`),
    with helper methods for converting between storage types and combining
    batches. Subclasses created via `make_dataclass` (see `ReplayBuffer._get_sample_class`)
    can add extra fields (e.g. tracked info like task ID).
    """
    observations: Union[np.ndarray, torch.Tensor]
    actions: Union[np.ndarray, torch.Tensor]
    next_observations: Union[np.ndarray, torch.Tensor]
    dones: Union[np.ndarray, torch.Tensor]
    rewards: Union[np.ndarray, torch.Tensor]

    def __post_init__(self):
        """Records the field names and storage type, validating the storage type."""
        self._fields = list(self.__dict__.keys())
        # Assumes attributes are always of the same storage type
        # either np.ndarray or torch.Tensor
        self._storage_type = type(self.observations)
        if self._storage_type not in (np.ndarray, torch.Tensor):
            raise TypeError(f"Storage type {self._storage_type} can only be np.ndarray or torch.Tensor")

    def __getitem__(self, index):
        """Indexes every field, returning a list of the indexed values (in field order)."""
        # return self.__class__(**{f: getattr(self, f)[index] for f in self._fields})
        return [getattr(self, f)[index] for f in self._fields]

    def __repr__(self):
        return f"{self.__class__.__name__}({self._storage_type})"

    def __len__(self):
        return len(self.observations)

    def to_tensor(self, device='cpu'):
        """Converts all fields to `torch.Tensor` (in place) and moves them to `device`.

        Args:
            device: Target torch device.

        Returns:
            self, with all fields converted/moved.

        Raises:
            TypeError: If the current storage type is neither `np.ndarray` nor
                `torch.Tensor`.
        """
        if self._storage_type is torch.Tensor:
            [setattr(self, f, getattr(self, f).to(device)) for f in self._fields]
            return self
        elif self._storage_type is not np.ndarray:
            raise TypeError(f"Samples storage {self._storage_type} is not of type np.ndarray")
        [setattr(self, f, torch.as_tensor(getattr(self, f)).to(device)) for f in self._fields]
        self._storage_type = torch.Tensor
        return self

    def to_numpy(self):
        """Converts all fields to `np.ndarray` (in place), detaching from the graph if needed.

        Returns:
            self, with all fields converted (or unchanged if already NumPy).

        Raises:
            TypeError: If the current storage type is neither `np.ndarray` nor
                `torch.Tensor`.
        """
        def get_numpy(f):
            f = getattr(self, f)
            return f.cpu().numpy() if not f.requires_grad else f.detach().cpu().numpy()
        if self._storage_type is np.ndarray:
            return
        elif self._storage_type is not torch.Tensor:
            raise TypeError(f"Samples storage {self._storage_type} is not of type torch.Tensor")
        [setattr(self, f, get_numpy(f)) for f in self._fields]
        self._storage_type = np.ndarray
        return self

    def concat(self, samples):
        """Stacks another `Samples` batch onto this one along the batch axis, in place.

        Args:
            samples: Another `Samples` instance with the same storage type.
                Only fields present in both `self` and `samples` are concatenated.

        Returns:
            self, with its shared fields extended by `samples`'s values.
        """
        assert self._storage_type is samples._storage_type
        def numpy_concat(f):
            self_f = getattr(self, f)
            samples_f = getattr(samples, f)
            return np.vstack([self_f, samples_f])

        def torch_concat(f):
            self_f = getattr(self, f)
            samples_f = getattr(samples, f)
            return torch.vstack([self_f, samples_f])

        if self._storage_type is np.ndarray:
            [setattr(self, f, numpy_concat(f)) for f in self._fields if f in samples._fields]
        else:
            [setattr(self, f, torch_concat(f)) for f in self._fields if f in samples._fields]
        return self

    def reduce_dims(self, dims, exclude=None, include=None):
        """Applies `crl.common.utils.reduce_dims` reshaping to some/all fields, in place.

        Args:
            dims: The `ordered_pairs` argument forwarded to `reduce_dims` for
                each field reshaped.
            exclude: If given, field names to skip (all others are reshaped).
                Mutually exclusive with `include`.
            include: If given, only these field names are reshaped. Mutually
                exclusive with `exclude`.

        Returns:
            self, with the selected fields reshaped.

        Raises:
            ValueError: If both `exclude` and `include` are given.
        """
        _reduce_dims_func = lambda f: setattr(self, f, reduce_dims(getattr(self, f), dims))
        if exclude is not None and include is not None:
            msg = "Can only specify arguments for exclude OR include not both."
            raise ValueError(msg)
        elif exclude is None and include is None:
            [_reduce_dims_func(f) for f in self._fields]
        elif exclude:
            [_reduce_dims_func(f) for f in self._fields if f not in exclude]
        elif include:
            [_reduce_dims_func(f) for f in self._fields if f in include]
        return self


class BaseBuffer(ABC):
    """Abstract base class for a rollout or replay buffer.

    Subclasses implement `_get_samples` to define how stored transitions are
    retrieved for a set of buffer indices; this class provides the shared
    bookkeeping (position, fullness, device conversion) and generic
    `sample`/`extend`/`reset` behavior built on top of it.
    """

    observation_space: spaces.Space
    obs_shape: Tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
    ):
        """
        Args:
            buffer_size: Max number of elements in the buffer.
            observation_space: Observation space of the environment.
            action_space: Action space of the environment.
            device: PyTorch device to which sampled values will be converted.
            n_envs: Number of parallel environments.
        """
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    def size(self) -> int:
        """Returns the current number of elements stored in the buffer."""
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """Adds a single transition to the buffer. Must be implemented by subclasses."""
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """Adds a batch of transitions to the buffer by calling `add` once per element.

        Args:
            *args: Batched transition arrays; each is iterated along its first
                (batch) axis and the corresponding per-element slices are passed
                to `add`.
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """Empties the buffer, resetting its position and fullness state."""
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env = None):
        """Randomly samples a batch of transitions from the buffer.

        Args:
            batch_size: Number of elements to sample.
            env: Associated gym `VecEnv`, used to normalize observations/rewards
                when sampling (forwarded to `_get_samples`).

        Returns:
            A `Samples` batch of size `batch_size`.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env = None
    ) -> Samples:
        """Retrieves stored transitions at the given buffer indices. Must be implemented by subclasses.

        Args:
            batch_inds: Array of buffer indices to retrieve.
            env: Associated gym `VecEnv`, used to normalize observations/rewards.

        Returns:
            A `Samples` batch containing the transitions at `batch_inds`.
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """Converts a NumPy array to a PyTorch tensor on this buffer's device.

        Args:
            array: The array to convert.
            copy: Whether to copy the data (default) rather than share memory
                with `array`. Inoperative if the device is not the CPU (a copy
                always happens when moving across devices).

        Returns:
            The array as a `torch.Tensor` on `self.device`.
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)


class ReplayBuffer(BaseBuffer):
    """Fixed-size NumPy-backed replay buffer used in off-policy algorithms like SAC/DQN.

    Stores observations, actions, rewards, dones (and optionally extra tracked
    info fields) in pre-allocated NumPy arrays, one slot per (timestep, env).
    Adapted from Stable-Baselines3's `ReplayBuffer`.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        keep_info_keys: dict = None,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        """
        Args:
            buffer_size: Max number of elements in the buffer (divided evenly
                across `n_envs`).
            observation_space: Observation space of the environment.
            action_space: Action space of the environment.
            keep_info_keys: Optional dict mapping extra info key -> a dict with
                'dims' (per-element shape) and 'type' (dtype), for additional
                per-transition data to store alongside the standard fields
                (e.g. task ID for rehearsal).
            device: PyTorch device to which sampled values will be converted.
            n_envs: Number of parallel environments.
            optimize_memory_usage: If True, stores next-observations by reusing
                the `observations` array (shifted by one slot) instead of a
                separate array, roughly halving memory use at the cost of extra
                indexing complexity. Cannot be combined with
                `handle_timeout_termination`. See
                https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
                and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274.
            handle_timeout_termination: If True, tracks which `done`s were due to
                a time limit (vs. a true episode end) so timeouts aren't treated
                as terminal transitions during sampling. See
                https://github.com/DLR-RM/stable-baselines3/issues/284.

        Raises:
            ValueError: If both `optimize_memory_usage` and
                `handle_timeout_termination` are True.
        """
        super().__init__(
            buffer_size=buffer_size, 
            observation_space=observation_space, 
            action_space=action_space, 
            device=device, 
            n_envs=n_envs
        )
 
        self.debug=False
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage
        
        # Shape (timesteps, envs, frames, color, height, width)
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        self.infos = {}
        self._track_info_keys = []
        if keep_info_keys is not None:
            self._track_info_keys = list(keep_info_keys.keys())
            for key, values in keep_info_keys.items():
                self.infos[key] = np.zeros((self.buffer_size, self.n_envs) + tuple(values['dims']), dtype=values['type'])
            
        self.sample_class = self._get_sample_class(self._track_info_keys)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes \
            + self.actions.nbytes \
            + self.rewards.nbytes \
            + self.dones.nbytes \
            + sum([i.nbytes for i in self.infos.values()])

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            print("Required Memory: {:.2f}GB Available Memory: {:.2f}GB".format(total_memory_usage/1e9, mem_available/1e9))
            if total_memory_usage > mem_available:
                total_memory_usage /= 1e9 # Convert to GB
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def __len__(self):
        return self.buffer_size if self.full else self.pos

    def _get_sample_class(self, add_attributes=None):
        """Builds the `Samples` subclass used to hold sampled batches, with any extra info fields.

        Args:
            add_attributes: Extra field names (from tracked info keys) to add
                to the dataclass, in addition to the base `Samples` fields.

        Returns:
            A dynamically created dataclass named 'ReplaySamples', subclassing `Samples`.
        """
        add_attributes = [] if add_attributes is None else add_attributes
        return make_dataclass('ReplaySamples', add_attributes, bases=(Samples,), repr=False)

    def extend(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        info: Dict[str, Any] = None,
    ) -> None:
        """Adds a batch of transitions to the buffer by calling `add` once per element.

        Args:
            obs: Batch of observations.
            next_obs: Batch of next observations.
            action: Batch of actions.
            reward: Batch of rewards.
            done: Batch of done flags.
            info: Optional batch (list) of per-transition info dicts.
        """
        for i in range(len(obs)):
            self.add(
                obs=obs[i],
                next_obs=next_obs[i],
                action=action[i],
                reward=reward[i],
                done=done[i],
                info=info[i] if info is not None else None
            )
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        info: Dict[str, Any] = None,
    ) -> None:
        """Adds a single transition (across all envs) to the buffer at the current position.

        Args:
            obs: Observation(s) for this timestep, one per env.
            next_obs: Next observation(s) for this timestep, one per env.
            action: Action(s) taken, one per env.
            reward: Reward(s) received, one per env.
            done: Done flag(s), one per env.
            info: Optional list of per-env info dicts; used to populate tracked
                info fields and (if `handle_timeout_termination`) the timeout flag.

        Raises:
            ValueError: If a tracked info key (from `keep_info_keys`) is missing
                from `info`.

        Note:
            `.copy()` seems unneeded as `np.shares_memory()` and
            `np.may_share_memory()` seem to always return False when not using
            `.copy()`. `np.array()` should always copy by default.
            Ref: https://github.com/DLR-RM/stable-baselines3/issues/112
        """
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        # Check info to see if there is any data that needs to be tracked.
        if len(self.infos) != 0:
            for key, _ in self.infos.items():
                data = []
                for env_idx, i in enumerate(info):
                    if key in i:
                        data.append(i[key])
                    else:
                        err = f"The key {key} was not detected in info. Make sure info " \
                            f"contains this key or remove it from keep_info_keys when " \
                            f"initializing ReplayBuffer."
                        raise ValueError(err)
                if len(data) != 0:
                    self.infos[key][self.pos] = np.array(data).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([i.get("TimeLimit.truncated", False) for i in info])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
            
    def sample_latest(self, batch_size: int, all_envs: bool = False, device: str = None):
        """Retrieves the most recently added `batch_size` transitions (not randomly sampled).

        Args:
            batch_size: Number of most-recent transitions to retrieve.
            all_envs: If True, keeps the per-env dimension distinct in the
                returned samples and then reduces it into the batch dimension;
                if False, samples come from whichever env slot they were stored in.
            device: Overrides the buffer's default device for the returned tensors.

        Returns:
            A `Samples` batch of the latest `batch_size` transitions, as tensors.

        Raises:
            ValueError: If `batch_size` is too large for the current buffer
                contents (accounting for `optimize_memory_usage`'s reserved slot).
        """
        device = device if device is not None else self.device
        
        # if not self.full and batch_size >= self.pos:
        #     raise ValueError("batch_size is greater or equal to current buffer size")
        if self.optimize_memory_usage:
            if len(self) - batch_size <= 0:
                err = f"batch_size {batch_size} can not be greater or equal to upper bound {len(self)} when taking the latest samples."
                raise ValueError(err)
        else:
            if len(self) - batch_size < 0:
                err = f"batch_size {batch_size} can not be greater to upper bound {len(self)} when taking the latest samples."
                raise ValueError(err)
            
        if self.full:
            # Will always wrap index, should not be considered until full
            batch_inds = (np.arange(self.buffer_size - batch_size, self.buffer_size) + self.pos) % self.buffer_size
        else:
            # Does not work if batch_size > self.pos, before full
            batch_inds = np.arange(self.pos - batch_size, self.pos)

        samples = self._get_samples(batch_inds, all_envs=all_envs)
            
        # Always reduce environment dimensions when all_env sampling is used
        if all_envs:
            samples.reduce_dims([[0, 1]])

        return samples.to_tensor(device)

    def sample(
        self,
        batch_size: int, 
        all_envs: bool = False,
        combined: bool = False,
        device: str = None
    ) -> Samples:
        """Randomly samples a batch of transitions from the replay buffer.

        Args:
            batch_size: Number of elements to sample from the buffer.
            all_envs: If True, samples from all environments (the last samples
                from every env are used); if False, randomly samples across
                environments for each drawn index.
            combined: If True, combines a random batch of `batch_size - 1`
                samples with the single most recent transition (always
                included). See https://arxiv.org/abs/1712.01275 for combined
                sampling theory.
            device: Overrides the buffer's default device for the returned tensors.

        Returns:
            A `Samples` batch of size `batch_size`, as tensors.

        Note:
            Uses custom index sampling when `optimize_memory_usage` is enabled,
            since the element at index `self.pos` must not be sampled (its
            "next observation" slot is being overwritten). See
            https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274.
        """
        device = device if device is not None else self.device
        
        if not self.optimize_memory_usage:
            # Buffer size and self.pos are not included!
            batch_inds = np.random.randint(0, len(self), size=batch_size)
        else:
            if self.full:
                # Start at 1 to prevent same index as self.pos from being drawn
                batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
            else:
                # End at self.pos to prevent same index as self.pos from being generated
                batch_inds = np.random.randint(0, self.pos, size=batch_size)
                
        # Replaces last index with the current sample index if it hasn't already been selected
        if combined:
            current_ind = (self.pos-1) % self.buffer_size
            if current_ind not in batch_inds:
                batch_inds[-1] = (self.pos-1) % self.buffer_size
        
        samples = self._get_samples(batch_inds, all_envs=all_envs)
      
        # Always reduce environment dimensions when all_env sampling is used
        if all_envs:
            samples.reduce_dims([[0, 1]])

        return samples.to_tensor(device)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        all_envs: bool = False,
    ) -> Samples:
        """Retrieves stored transitions at the given buffer indices.

        Args:
            batch_inds: Array of buffer indices to retrieve.
            all_envs: If True, retrieves data for every env at each index
                (`env_inds` becomes a full slice); if False, picks one random
                env per index.

        Returns:
            A `Samples` (or `ReplaySamples`) batch containing the transitions
            at `batch_inds`, including any tracked info fields.
        """
        if all_envs:
            env_inds = slice(None)
        else:
            env_inds = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_inds, :]
        else:
            next_obs = self.next_observations[batch_inds, env_inds, :]

        data = (
            self.observations[batch_inds, env_inds, :],
            self.actions[batch_inds, env_inds, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_inds] * (1 - self.timeouts[batch_inds, env_inds]))[..., None],
            self.rewards[batch_inds, env_inds][..., None],
        )
        # Append tracked info data, if empty, nothing will happen
        data += tuple([info_array[batch_inds, env_inds] for _, info_array in self.infos.items()])

        return self.sample_class(*data)