import warnings
from dataclasses import make_dataclass
from typing import Callable, Union, Dict
from pdb import set_trace

import numpy as np
import torch
from gym import spaces     

from crl.common.buffers.replay_buffers import Samples, ReplayBuffer


class RehearsalReplayBuffer(ReplayBuffer):
    """A `ReplayBuffer` holding a curated set of rehearsal samples with updatable tracked info.

    Unlike the standard replay buffer (which stores whatever transitions the
    agent just experienced), this buffer holds a selected subset of past
    samples (see `RRBManager.add_rehearsal_samples`) whose tracked info fields
    (e.g. cached model outputs) can later be refreshed in place via `update`.
    Always uses a single environment and disables memory-usage optimization and
    timeout handling, since samples are not sequential.
    """

    def __init__(
        self,
        output_info: dict,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        track_task_id: bool = True,
        update_method: str = 'replace',
    ):
        """
        Args:
            output_info: Dict describing extra tracked info fields to store per
                sample (passed through as `keep_info_keys` to `ReplayBuffer`),
                e.g. cached model outputs to rehearse on.
            buffer_size: Max number of rehearsal samples to store.
            observation_space: Observation space of the environment.
            action_space: Action space of the environment.
            device: PyTorch device to which sampled values will be converted.
            track_task_id: If True, expects/stores a 'task_id' tracked info
                field, enabling `get_task_id_locs`.
            update_method: How `update` combines old and new tracked-info
                values: 'replace' overwrites, 'average' averages old and new.

        Raises:
            ValueError: If `update_method` is not 'replace' or 'average'.
        """
        self.track_task_id = track_task_id

        super().__init__(
            buffer_size=buffer_size, 
            observation_space=observation_space, 
            action_space=action_space, 
            device=device, 
            n_envs=1, # can never be more than one for now
            optimize_memory_usage=False, # can never be true as samples are not sequential
            handle_timeout_termination=False, # can never be true as samples are not sequential
            keep_info_keys=output_info,
        )

        if update_method == 'replace':
            self.update_method = self._replace
        elif update_method == 'average':
            self.update_method = self._average
        else:
            msg = f"`update` must be 'replace' or 'average', got {update_method}"
            raise ValueError(msg)
            
    def _get_sample_class(self, add_attributes=None):
        """Builds the `Samples` subclass used to hold sampled batches, with any extra info fields.

        Args:
            add_attributes: Extra field names (from tracked info keys) to add
                to the dataclass, in addition to the base `Samples` fields.

        Returns:
            A dynamically created dataclass named 'RelevantSamples', subclassing `Samples`.
        """
        add_attributes = [] if add_attributes is None else add_attributes
        return make_dataclass('RelevantSamples', add_attributes, bases=(Samples,), repr=False)

    def update(
        self,
        info_name,
        value,
        index: int = slice(None),
    ):
        """Updates a tracked info field in place using `self.update_method` ('replace' or 'average').

        Args:
            info_name: Name of the tracked info field to update.
            value: New value(s) to combine with the existing value(s) at `index`.
            index: Index or slice selecting which stored entries to update.
                Defaults to updating all entries.

        Note:
            Emits a warning if the updated value ends up identical to the old
            value, which may indicate the update did not actually change anything.
        """
        old_value = self.infos[info_name][index].copy()
        new_value = self.update_method(self.infos[info_name], value, index)
        if np.all(old_value == new_value):
            msg = f"{info_name} might not have been updated!"
            print(f"Old value:\n{old_value}")
            print(f"New value:\n{new_value}")
            warnings.warn(msg)

    def _replace(self, array, value, index):
        """Overwrites `array[index]` with `value`.

        Args:
            array: The tracked-info array to update.
            value: New value(s), must match `array[index]` in length.
            index: Index or slice into `array`.

        Returns:
            The updated `array[index]`.
        """
        assert len(array[index]) == len(value), f"{len(array[index])} != {len(value)}"
        array[index] = value
        return array[index]

    def _average(self, array, value, index):
        """Averages `array[index]` with `value`, in place.

        Args:
            array: The tracked-info array to update.
            value: New value(s), must match `array[index]` in length.
            index: Index or slice into `array`.

        Returns:
            The updated `array[index]`.
        """
        assert len(array[index]) == len(value), f"{len(array[index])} != {len(value)}"
        array[index] = (value + array[index]) / 2
        return array[index]

    def get_task_id_locs(self):
        """Gets the buffer indices belonging to each unique tracked task ID.

        Returns:
            Dict mapping unique task ID to an array of buffer indices holding
            samples from that task, if `track_task_id` is enabled; otherwise an
            empty array.
        """
        if self.track_task_id:
            task_id = self.infos['task_id'][:len(self)]
            unique_ids = np.unique(task_id)
            id_locs = {uid: np.where(task_id == uid)[0] for uid in unique_ids}
            return id_locs
        
        return np.array([])
    
    
class RRBManager():
    """Coordinates a standard replay buffer and a rehearsal replay buffer together.

    Provides joint sampling from both buffers, and manages populating/refreshing
    the rehearsal buffer with a curated subset of past samples (plus cached
    model outputs computed via `get_module_outputs_fn`) for continual-learning
    rehearsal.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        rehearsal_replay_buffer: RehearsalReplayBuffer,
        get_module_outputs_fn: Callable,

    ):
        """
        Args:
            replay_buffer: The standard (task-local) replay buffer.
            rehearsal_replay_buffer: The rehearsal buffer holding samples curated
                from `replay_buffer`'s history.
            get_module_outputs_fn: Callable that takes a batch of observations
                and returns a dict of named model outputs (e.g. embeddings,
                Q-values) to cache alongside rehearsal samples.
        """
        self.rb = replay_buffer
        self.rrb = rehearsal_replay_buffer
        self.get_module_outputs_fn = get_module_outputs_fn

    def sample(
        self,
        batch_size: int,
        rrb_batch_size: int,
        combined: bool =  False,
        device: str = None,
    ):
        """Samples from the standard buffer and the rehearsal buffer independently.

        Args:
            batch_size: Number of samples to draw from the standard replay
                buffer; if 0 (or the buffer is empty), no standard samples are drawn.
            rrb_batch_size: Number of samples to draw from the rehearsal
                buffer; if 0 (or the buffer is empty), no rehearsal samples are drawn.
            combined: Forwarded to the standard buffer's `sample` (combined
                sampling); not applied to the rehearsal buffer.
            device: Overrides each buffer's default device for the returned tensors.

        Returns:
            Tuple `(replay_samples, rel_samples)`, either of which may be None
            if its respective batch size was 0 or its buffer was empty.
        """
        if len(self.rb) == 0 or batch_size == 0 :
            replay_samples = None
        else:
            replay_samples = self.rb.sample(
                batch_size=batch_size,
                combined=combined,
                device=device,
            )
            
        if len(self.rrb) == 0 or rrb_batch_size == 0:
            rel_samples = None
        else:
            rel_samples = self.rrb.sample(
                batch_size=rrb_batch_size,
                combined=False,
                device=device,
            )
        
        return replay_samples, rel_samples
    
    def add_rehearsal_samples(
        self, 
        replay_history_length: int,
        sample_size: int,
        iteratively: bool = False, 
        iter_size: int = 100,

    ):
        """Selects a random subset of recent samples and stores them in the rehearsal buffer.

        Draws `sample_size` samples from the last `replay_history_length`
        transitions in the standard replay buffer, computes cached model
        outputs for their observations (e.g. embeddings/Q-values), and adds
        them to the rehearsal buffer along with those outputs (and task ID, if tracked).

        Args:
            replay_history_length: The number of latest samples to draw from
                in the standard replay buffer when selecting rehearsal
                candidates. If None, uses all but the most recent sample.
            sample_size: Number of samples to select from that history.
            iteratively: If True, computes model outputs in chunks (see
                `_iteratively_compute_relevant`) instead of all at once. Useful
                when the candidate observations don't fit into GPU memory at once.
            iter_size: Chunk size used when `iteratively` is True.

        Returns:
            The samples that were added to the rehearsal replay buffer.
        """
        # NOTE: If using multi environments then the returned samples can be
        #       greater than replay_history_length.
        if replay_history_length is None:
            replay_history_length = len(self.rb)-1

        replay_samples = self.rb.sample_latest(
            batch_size=replay_history_length, 
            all_envs=True,
            device='cpu' if iteratively else None
        )
        assert len(replay_samples) == (replay_history_length * self.rb.n_envs)

        rand_replay_idx = np.random.choice(
            np.arange(len(replay_samples)), 
            size=sample_size, 
            replace=False
        )
        
        obs = replay_samples.observations[rand_replay_idx]
        if iteratively:
            model_outputs = self._iteratively_compute_relevant(
                observations=obs, 
                iter_size=iter_size
            )
        else:
            with torch.no_grad():
                model_outputs = self.get_module_outputs_fn(obs)
            # Convert outputs to NumPy arrays
            for name, output in model_outputs.items():
                model_outputs[name] = output.cpu().numpy()    
        replay_samples.to_numpy()

        # NOTE: Since we are hijacking info for storage, we need to separate the arrays
        #       into a list of dictionaries to match expected format for info.
        info = []
        for i, rand_idx in enumerate(rand_replay_idx):
            info_dict = {}
            for name, output in model_outputs.items():
                info_dict[name] = output[i, None], # Add n_envs empty dimension 
            if self.rrb.track_task_id:
                info_dict['task_id'] = replay_samples.task_id[rand_idx]
            info.append([info_dict])

        self.rrb.extend(
            obs=replay_samples.observations[rand_replay_idx, None], # Add n_envs empty dimension 
            action=replay_samples.actions[rand_replay_idx, None], # Add n_envs empty dimension 
            next_obs=replay_samples.next_observations[rand_replay_idx, None], # Add n_envs empty dimension 
            done=replay_samples.dones[rand_replay_idx],
            reward=replay_samples.rewards[rand_replay_idx],
            info=info
        )

        return self.rrb.sample_latest(len(rand_replay_idx))

    def update_rehearsal_samples(self, task_id: Union[str, int], iter_size: int = 100):
        """Recomputes and refreshes cached model outputs for a task's stored rehearsal samples.

        Args:
            task_id: The task ID whose rehearsal samples should be refreshed.
            iter_size: Chunk size used when recomputing model outputs (see
                `_iteratively_compute_relevant`).

        Raises:
            ValueError: If the rehearsal buffer is not tracking task IDs.

        Note:
            This updates ALL of a task's cached outputs (e.g. Q-values for every
            action) regardless of whether the task actually uses all of them
            (i.e., only a subset of the actions).
        """
        if len(self.rrb) == 0:
            return
        if not self.rrb.track_task_id:
            msg = "Can not update samples without tracking task IDs."
            raise ValueError(msg)

        id_locs = self.rrb.get_task_id_locs()
        if task_id not in id_locs:
            print(f"DEBUG: NO TASK ID {task_id} FOUND!")
            return
        task_id_locs = id_locs[task_id]
        observations = torch.tensor(self.rrb.observations[task_id_locs].squeeze(axis=1))

        model_outputs = self._iteratively_compute_relevant(
            observations=observations,
            iter_size=iter_size
        )
        
        for name, output in model_outputs.items():
            self.rrb.update(
                info_name=name,
                value=output[:, None],
                index=task_id_locs,
            )

    def _iteratively_compute_relevant(
        self,
        observations: torch.Tensor,
        iter_size: int = 100
    ) -> Dict[str, np.ndarray]:
        """Computes `get_module_outputs_fn` outputs for `observations` in chunks, to bound memory use.

        Args:
            observations: The observations to compute model outputs for.
            iter_size: Number of observations to process per chunk.

        Returns:
            Dict mapping output name to a NumPy array concatenated across all
            chunks, in the original observation order.
        """
        model_outputs = {}
        for i in range(0, len(observations), iter_size):
            with torch.no_grad():
                output_dict = self.get_module_outputs_fn(observations[i:i+iter_size])
                assert isinstance(output_dict, dict)
            
            for name, output in output_dict.items():
                if name not in model_outputs:
                    model_outputs[name] = [output.cpu().numpy()]
                else:
                    model_outputs[name].append(output.cpu().numpy())

        for name, output in model_outputs.items():
            model_outputs[name] = np.vstack(output)
            
        return model_outputs