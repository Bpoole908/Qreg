import os
from pdb import set_trace

import numpy as np

from crl.policies.dqn.dqn_policy import DQNPolicy
from crl.policies.data_rehearsal.drdqn_policy_config import DRDQNPolicyConfig
from crl.common.buffers.rehearsal_replay_buffer import RRBManager, RehearsalReplayBuffer
from crl.common.buffers.replay_buffers import ReplayBuffer
from crl.common.utils import rgetattr
from crl.policies.data_rehearsal.drdqn_loss import DRDQNLoss
from crl.common.utils import pickle_dump, pickle_load


class DRDQNPolicy(DQNPolicy):
    """DQN with "qreg" data-rehearsal regularization: rehearses cached embeddings/Q-values instead of raw past transitions.

    Maintains a `RehearsalReplayBuffer` of curated past observations with
    cached DQN embeddings/Q-values, and regularizes the DQN towards those
    cached outputs (`DRDQNLoss`) during training, mitigating forgetting
    without replaying full past transitions.
    """

    def __init__(self, config: DRDQNPolicyConfig, observation_space, action_spaces):
        """
        Args:
            config: Policy configuration, including data-rehearsal-specific
                hyperparameters (`rrb_buffer_size`, `rrb_batch_size`,
                `rrb_add_frequency`, `rrb_update_frequency`,
                `rrb_loss_kwargs`, etc.).
            observation_space: Gym observation space, shared across tasks.
            action_spaces: Dict mapping action_space_id to a Gym action space;
                all must have the same `.shape`.
        """
        super(DRDQNPolicy, self).__init__(config, observation_space, action_spaces)
        self.rrb_loss = DRDQNLoss(**self.cfg.rrb_loss_kwargs)
        self.wait_to_reg = self.cfg.wait_to_reg # If false, No-Wait
        self.rrb_add_frequency = self._validate_rrb_frequency(self.cfg.rrb_add_frequency)
        self.rrb_update_frequency = self._validate_rrb_frequency(self.cfg.rrb_update_frequency)

    def _validate_rrb_frequency(self, rrb_rate):
        """Validates (and if needed, rounds up) a rehearsal add/update frequency so no iteration is skipped.

        Args:
            rrb_rate: The configured frequency (in timesteps), or None to
                leave rehearsal add/update disabled.

        Returns:
            None if `rrb_rate` is None; otherwise `rrb_rate` unchanged, or the
            next multiple of `n_envs` above it if the configured value would
            cause skipped iterations.
        """
        if rrb_rate is None:
            return None

        remainder = rrb_rate % self.n_envs
        # If remainder does not go evenly into frequency then it will skip some
        # iterations. Thus, compute the next closest frequency.
        if remainder != 0:
            new_rrb_rate = int(np.ceil(rrb_rate/self.n_envs)*self.n_envs)
            msg = f"The rehearsal sample rate of {rrb_rate} will have skips. Using {new_rrb_rate} instead."
            self.logger.warn(msg)
            assert new_rrb_rate % self.n_envs == 0
            return new_rrb_rate
        return rrb_rate

    def init_replay_buffer(self):
        """Builds the standard replay buffer, the rehearsal replay buffer, and their `RRBManager`.

        The rehearsal buffer tracks cached DQN embeddings/Q-values
        (`embed`, `q_values`) alongside each observation, plus task ID if configured.

        Note:
            If the below tracked field names are changed, `_get_model_outputs`
            and `DRDQNLoss` must be updated to match.
        """
        output_info = dict(
            q_values=dict(dims=(self.action_space.n,), type='float'),
            embed=dict(dims=(self.dqn.embedding_size,), type='float'),
        )
        
        if self.cfg.track_task_id:
            task_id = dict(dims=tuple(), type=np.int16)
            # Makes sure RE is tracking task ID for each sample
            keep_info_keys = dict(task_id=task_id)
            # Makes sure RRB is tracking task ID
            output_info['task_id'] = task_id
            
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.cfg.buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            keep_info_keys=keep_info_keys,
            n_envs=self.n_envs,
            device=self.device,
            handle_timeout_termination=self.cfg.handle_timeout_termination,
            optimize_memory_usage=self.cfg.optimize_memory_usage,
        )

        self.rehearsal_replay_buffer = RehearsalReplayBuffer(
            output_info=output_info,
            buffer_size=self.cfg.rrb_buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            update_method='replace'
        )

        self.rrb_manager = RRBManager(
            replay_buffer=self.replay_buffer,
            rehearsal_replay_buffer=self.rehearsal_replay_buffer,
            get_module_outputs_fn=self._get_model_outputs,
        )
        
    def _compute_training_action(self, observation, task_id, action_space, last_timestep_data):
        """Selects a training action (epsilon-greedy) and manages the replay/rehearsal buffers.

        Args:
            observation: Current observation from the environment.
            task_id: Current task id, typically an int.
            action_space: Current action space.
            last_timestep_data: Data about the prior timestep, or None if a
                task just switched.

        Returns:
            The selected action tensor.

        Note:
            Overrides `DQNPolicy._compute_training_action`. In order: 1)
            appends `task_id` to info so the rehearsal buffer can use it, if
            enabled; 2) computes a greedy or random (epsilon-greedy) action;
            3) appends the last stored transition, if any, to the replay
            buffer; 4) updates the target DQN if it's due; 5) periodically
            refreshes cached rehearsal-sample outputs
            (`rrb_update_frequency`); 6) periodically adds new rehearsal
            samples (`rrb_add_frequency`) — must come after appending to the
            replay buffer, or an error is raised for not having enough samples
            when `rrb_add_history <= rrb_add_frequency`; 7) checks whether
            `task_id` is new and, if so, updates `self.task_id` (done last
            since `last_timestep_data` corresponds to the task_id before this update).
        """
        # Add task_id to info if it is being tracked by replay buffer
        # last_timestep_data will always be none when tasks switch.
        if self.cfg.track_task_id and last_timestep_data is not None:
            for env_info in last_timestep_data.info:
                env_info['task_id'] = task_id
        
        # NOTE: exploration_rate is NOT annealed over time currently
        if self.cfg.exploration_rate != 0 and np.random.rand() < self.cfg.exploration_rate:
            action = self._get_random_action(action_space)
        else:
            action = self._get_greedy_action(observation, action_space)

        if last_timestep_data is not None:
            self.replay_buffer.add(
                obs=last_timestep_data.observation.numpy(),
                next_obs=observation.numpy(),
                action=last_timestep_data.action.numpy(), 
                reward=last_timestep_data.reward, 
                done=last_timestep_data.done,
                info=last_timestep_data.info
            )
        
        # Update timestep counter very last as these actions are for the coming timesteps
        if self.total_timesteps > 0 and self.total_timesteps % self.target_update_frequency == 0:
            self.logger.info(f"{self.total_timesteps} Updating target model")
            self._update_target_model()
        
        # UPDATE RRB samples
        update_condi = (
            self.rrb_update_frequency
            and len(self.rehearsal_replay_buffer) > 0
            and self.total_timesteps > 0
            and self.total_timesteps % self.rrb_update_frequency == 0
        )
        if update_condi:
            self._update_rrb_samples()
            
        # ADD new RRB samples
        add_condi = (
            self.total_timesteps > 0 # Not first timestep
            and self.total_timesteps % self.cfg.rrb_add_frequency == 0 # Time to add
        )
        if add_condi:
            self._add_rrb_samples()
                
        # NOTE: This must come AFTER adding rehearsal samples and updating them, otherwise if
        #       rrb_update_frequency falls on the same step the task changes, 
        #       then wrong samples will be updated.
        if self.wait_to_reg and self.task_id is not None and self.task_id != task_id:
            self.wait_to_reg = False
            
        self._update_task_info(task_id, action_space)

        return action

    def _update_rrb_samples(self):
        """Refreshes cached DQN embeddings/Q-values for the current task's stored rehearsal samples.

        Note:
            Torch eval mode is not manually enabled, since these cached sample
            values should reflect what is generated during training.
        """
        self.logger.info(f"{self.total_timesteps}: Updating rehearsal samples for task {self.task_id}")
        self.rrb_manager.update_rehearsal_samples(
            task_id=self.task_id,
            iter_size=self.cfg.iterative_compute_size
        )
    
    def _add_rrb_samples(self):
        """Selects and adds new samples (with cached DQN embeddings/Q-values) to the rehearsal replay buffer.

        Note:
            Torch eval mode is not manually enabled, since these cached sample
            values should reflect what is generated during training.
        """
        self.logger.info(f"{self.total_timesteps}: Gathering rehearsal samples during task {self.task_id}")
        rehearsal_samples = self.rrb_manager.add_rehearsal_samples(
            replay_history_length=self.cfg.rrb_add_history,
            sample_size=self.cfg.rrb_add_sample_size,
            iteratively=self.cfg.iteratively_compute_samples,
            iter_size=self.cfg.iterative_compute_size,
        )
        self.logger.info(f"{self.total_timesteps}: Gathered {len(rehearsal_samples)} rehearsal samples")
        self.logger.info(f"{self.total_timesteps}: Rehearsal buffer size: {len(self.rehearsal_replay_buffer)}, Pos: {self.rehearsal_replay_buffer.pos}")
        self.logger.info(f"{self.total_timesteps}: Buffer size: {len(self.replay_buffer)}, Pos: {self.replay_buffer.pos}")
        if self.rehearsal_replay_buffer.track_task_id:
            id_locs = self.rehearsal_replay_buffer.get_task_id_locs()
            self.logger.info(f"{self.total_timesteps}: Samples per ID {[ f'{k}: {len(v)}' for k, v in id_locs.items()]}")
        
    def train(self, storage_buffer):
        """Performs one gradient step on a joint replay + rehearsal minibatch, if enough replay data exists.

        Args:
            storage_buffer: Samples collected since `train()` was last called.
                Unused here (data was already added to the replay buffer in
                `_compute_training_action`); generated by the `EnvironmentRunner`.

        Returns:
            The Tensorboard-style log list from the loss computation (with
            weight/gradient histograms occasionally appended, per
            `cfg.log_sample_rate`), or None if there weren't yet enough
            samples in the replay buffer to train.
        """
        if not self.replay_buffer.full and self.cfg.update_after >= self.replay_buffer.pos:
            self.logger.debug(f"{self.total_timesteps} Only {len(self.replay_buffer)} samples, need {self.cfg.update_after} samples before training")
            return
        # Determine batch size for reg samples drawn from RRB
        if self.wait_to_reg:
            rrb_batch_size = 0
        elif len(self.rehearsal_replay_buffer) < self.cfg.rrb_batch_size:
            rrb_batch_size = len(self.rehearsal_replay_buffer)
        else:
            rrb_batch_size = self.cfg.rrb_batch_size
        
        # Randomly select times to log to save space
        log_sample = np.random.rand() < self.cfg.log_sample_rate
        
        replay_samples, rehearsal_samples = self.rrb_manager.sample(
            batch_size=self.cfg.batch_size,
            rrb_batch_size=rrb_batch_size,
            device=self.device
        )

        # Compute Losses
        loss, logs = self._compute_loss(replay_samples, rehearsal_samples, log_sample) 
        
        # Run optimization and backprop
        self.optimizer.zero_grad()
        loss.backward()
        # Randomly log 
        if log_sample and self.cfg.networks_to_log is not None:
            logs += self._log_weights_and_grads(self.dqn)
        self.optimizer.step()
        
        return logs

    def _log_weights_and_grads(self, model):
        """Builds Tensorboard histogram log entries for the weights/gradients of configured sub-networks.

        Args:
            model: The model (e.g. `self.dqn`) whose named sub-networks
                (per `cfg.networks_to_log`) will be logged.

        Returns:
            A list of Tensorboard-style histogram log dicts, one per
            parameter (weights, plus gradients where available).
        """
        logs = []
        for network_name in self.cfg.networks_to_log:
            network = rgetattr(model, network_name)
            for n, p in network.named_parameters():
                n = f"{network_name}.{n}"
                logs.append({"type": "histogram", "tag": f"params/{n}", "value": p.cpu()})
                if hasattr(p, 'grad') and p.grad is not None:
                    logs.append({"type": "histogram", "tag": f"grads/{n}", "value": p.grad.cpu()})
        return logs

    def _compute_loss(self, replay_samples, rehearsal_samples=None, log=False):
        """Computes the DQN replay loss plus the qreg rehearsal regularization penalty.

        Args:
            replay_samples: A `Samples` batch drawn from the replay buffer.
            rehearsal_samples: A `Samples` batch drawn from the rehearsal
                replay buffer, or None if rehearsal isn't yet active.
            log: If True, builds the full Tensorboard-style log list;
                otherwise returns an empty log (to save space/computation).

        Returns:
            Tuple `(loss, logs)`: the combined loss (replay + rehearsal
            penalty), and (if `log`) a list of Tensorboard-style log dicts
            including the total, replay-only, and rehearsal-only loss components.

        Raises:
            AssertionError: If `wait_to_reg` is set but `rehearsal_samples` is
                not None, or if the rehearsal buffer is non-empty but
                `rehearsal_samples` is None.
        """
        logs, rrb_log = [], []
        rrb_loss = 0 

        replay_loss, replay_log = self.loss(
            dqn=self.dqn,
            target_dqn=self.target_dqn,
            replay_samples=replay_samples,
            **self.cfg.loss_kwargs
        )
        
        # TODO: DEBUG REMOVE
        if self.wait_to_reg:
            assert rehearsal_samples is None 
        elif len(self.rehearsal_replay_buffer) != 0:
            assert rehearsal_samples is not None 

        # Apply Qreg loss (Eq. 1 from paper)
        if rehearsal_samples is not None:
            rrb_loss, rrb_log = self.rrb_loss(
                dqn=self.dqn,
                rehearsal_samples=rehearsal_samples,
            )
        loss = replay_loss + rrb_loss
        if log:
            logs = [
                {"type": "scalar", "tag": "loss", "value": loss},
                {"type": "scalar", "tag": "replay_loss", "value": replay_loss},
                {"type": "scalar", "tag": "rrb_loss", "value": rrb_loss}
            ] + rrb_log + replay_log
        
        return loss, logs
    
    def _get_model_outputs(self, observations):
        """Computes DQN embeddings and Q-values for a batch of observations, for rehearsal caching.

        Args:
            observations: Batch of observations to compute outputs for.

        Returns:
            Dict with keys 'embed' and 'q_values', matching the tracked info
            fields set up in `init_replay_buffer`.
        """
        obs = observations.to(self.device)
        embed, q_values = self.dqn(obs)

        return dict(embed=embed, q_values=q_values)

    def save(self, output_path_dir, cycle_id, task_id, task_total_steps):
        """Saves the DQN/target DQN/optimizer state (via `DQNPolicy.save`) plus the rehearsal replay buffer.

        Args:
            output_path_dir: Directory to save 'dqn.pt' and (if configured)
                'replay_buffer.pkl'/'rehearsal_replay_buffer.pkl' into.
            cycle_id: Current cycle. Unused, kept for `PolicyBase` API compatibility.
            task_id: Current task id. Unused, kept for `PolicyBase` API compatibility.
            task_total_steps: Current number of total steps. Unused, kept for
                `PolicyBase` API compatibility.

        Note:
            Loading still requires constructing the policy with its required
            arguments; `load` then overwrites the model and optimizer states in place.
        """
        super().save(output_path_dir, cycle_id, task_id, task_total_steps)

        if self.cfg.save_replay_buffer:
            rrb_replay_path = os.path.join(output_path_dir, 'rehearsal_replay_buffer.pkl')
            pickle_dump(self.rehearsal_replay_buffer, rrb_replay_path)

    def load(self, output_path_dir):
        """Loads the DQN/target DQN/optimizer/replay-buffer state (via `DQNPolicy.load`) plus the rehearsal replay buffer.

        Args:
            output_path_dir: Directory containing a previously saved 'dqn.pt'
                and (optionally) 'replay_buffer.pkl'/'rehearsal_replay_buffer.pkl',
                as written by `save`. Rebuilds `self.rrb_manager` afterwards so
                it references the (possibly newly loaded) buffers.
        """
        super().load(output_path_dir)
        rrb_replay_path = os.path.join(output_path_dir, 'rehearsal_replay_buffer.pkl')
        if os.path.exists(rrb_replay_path):
            self.logger.info(f"Loading reg replay buffer: {rrb_replay_path}") 
            self.rehearsal_replay_buffer = pickle_load(rrb_replay_path)
            
        self.rrb_manager = RRBManager(
            replay_buffer=self.replay_buffer,
            rehearsal_replay_buffer=self.rehearsal_replay_buffer,
            get_module_outputs_fn=self._get_model_outputs,
        )