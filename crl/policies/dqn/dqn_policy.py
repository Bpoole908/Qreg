import os
from typing import Union
from pdb import set_trace

import numpy as np
import torch
from gym.spaces import Space
from torchviz import make_dot
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch
from continual_rl.utils.utils import Utils

from crl.common.buffers.replay_buffers import ReplayBuffer
from crl.common.utils import pickle_dump, pickle_load
from crl.policies.dqn.model import DQN
from crl.policies.dqn.dqn_losses import DQNLosses
from crl.policies.dqn.dqn_policy_config import DQNPolicyConfig
from crl.policies.dqn.dqn_timestep_data import DQNTimestepData


class DQNPolicy(PolicyBase):
    """DQN policy (Mnih et al. 2015) for discrete action spaces, integrated with `continual_rl`.

    Wraps a `DQN` model, a target network, a `ReplayBuffer`, and a `DQNLosses`
    loss function into a `continual_rl` `PolicyBase` implementation: it selects
    actions (`compute_action`), stores transitions and periodically syncs the
    target network during training, and performs gradient steps (`train`) on
    minibatches sampled from the replay buffer. Assumes all tasks share the
    same action space shape.
    """

    def __init__(self, config: DQNPolicyConfig, observation_space, action_spaces):
        """
        Args:
            config: Policy configuration (hyperparameters, buffer settings, etc.).
            observation_space: Gym observation space, shared across tasks.
            action_spaces: Dict mapping action_space_id to a Gym action space;
                all must have the same `.shape`.

        Raises:
            AssertionError: If the action spaces don't all share the same shape.
        """
        super(DQNPolicy, self).__init__(config)
        self.cfg = config
        self.logger = Utils.create_logger(f"{config.output_dir}/policy.log")
        self.observation_space = observation_space
        self.action_spaces = action_spaces
        action_spaces = list(action_spaces.values())
        self.action_space = action_spaces[0]
        for act_space in action_spaces[1:]:
            assert self.action_space.shape == act_space.shape, "DQN only supports environments with same action spaces."
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_envs = 1 
        self.task_id = None # Used for tracking task change
        self.total_timesteps = 0 # Tracks total timesteps across all tasks
        self.target_update_frequency = self._validate_target_update_frequency()
        
        # Build models and optimizers
        self.init_models()
        # Build replay buffer after models in case it needs knowledge about models 
        self.init_replay_buffer()
        self.init_model_optimizers()
        # Build loss
        self.loss = self.get_loss()
        # Visualize models
        self.visualize_models()
  
    def visualize_models(self):
        """Renders the DQN's embedding and Q-value computation graphs to PNG files, for debugging."""
        X = torch.Tensor(self.observation_space.sample()[None]).to(self.device)
        embed, q_values = self.dqn(X)
        vis = make_dot(q_values, params=dict(self.dqn.named_parameters()))
        vis.render('dqn-q-values', format='png')
        vis = make_dot(embed, params=dict(self.dqn.named_parameters()))
        vis.render('dqn-embedding', format='png')
        
    def _validate_target_update_frequency(self):
        """Validates (and if needed, rounds up) `target_update_frequency` so no update is skipped.

        Returns:
            The configured `target_update_frequency`, or the next multiple of
            `n_envs` above it if the configured value would cause skipped updates.

        Note:
            An update is considered skipped if `target_update_frequency %
            n_envs` is not 0. Not really needed when the number of environments
            is locked to 1, as it currently is.
        """
        tuf = self.cfg.target_update_frequency
        remainder = tuf % self.n_envs
        # If remainder does not go evenly into update frequency then it will skip some
        # update iterations. Thus, compute the next closest update frequency.
        if remainder != 0:
            new_tuf= int(np.ceil(tuf/self.n_envs)*self.n_envs)
            msg = f"Target update frequency {tuf} will have skips. Using {new_tuf} instead."
            self.logger.warn(msg)
            assert new_tuf % self.n_envs == 0
            return new_tuf
        
        return tuf

    def init_models(self):
        """Builds the online `DQN` and target `DQN` networks and syncs the target to the online weights.
        """
        n_actions = self.action_space.n
        # Base Model
        self.dqn = DQN(
            observation_space=self.observation_space, 
            output_dims=n_actions,
            **self.cfg.dqn_kwargs
        ).to(self.device)
        # Target Model
        self.target_dqn = DQN(
            observation_space=self.observation_space, 
            output_dims=n_actions,
            **self.cfg.dqn_kwargs
        ).to(self.device)
        self._update_target_model()
    
    def _update_target_model(self):
        """Copies the online DQN's weights into the target DQN."""
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def init_model_optimizers(self) -> None:
        """Initializes (or re-initializes) the DQN's optimizer using the configured class/learning rate."""
        self.optimizer = self.cfg.optimizer_class(self.dqn.parameters(), lr=self.cfg.learning_rate)

    def get_loss(self):
        """Looks up the configured loss function on `DQNLosses` by name.

        Returns:
            The static loss method named `self.cfg.loss_name` on `DQNLosses`.

        Raises:
            ValueError: If `DQNLosses` has no method named `self.cfg.loss_name`.
        """
        if hasattr(DQNLosses, self.cfg.loss_name):
            return getattr(DQNLosses, self.cfg.loss_name)
        else:
            msg = f"DQNLosses has no loss method {self.cfg.loss_name}"
            raise ValueError(msg)
    
    def init_replay_buffer(self):
        """Builds `self.replay_buffer`, tracking task ID as extra info if configured to."""
        keep_info_keys = {}
        if self.cfg.track_task_id:
            task_id = dict(dims=tuple(), type=np.int16)
            # Makes sure RB is tracking task ID for each sample
            keep_info_keys = dict(task_id=task_id)
            
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
    
    def get_environment_runner(self, task_spec):
        """Builds the `continual_rl` environment runner used to collect experience for this policy.

        Args:
            task_spec: Unused; accepted for `PolicyBase` API compatibility.

        Returns:
            An `EnvironmentRunnerBatch` configured with this policy's collection settings.
        """
        runner = EnvironmentRunnerBatch(
            policy=self, 
            num_parallel_envs=self.n_envs,
            timesteps_per_collection=self.cfg.timesteps_per_collection,
            render_collection_freq=self.cfg.render_collection_freq,
            output_dir=self.cfg.output_dir
        )
        return runner

    def compute_action(
        self, 
        observation: torch.Tensor, 
        task_id: Union[int, str], 
        action_space_id: Union[int, str], 
        last_timestep_data: DQNTimestepData, 
        eval_mode: bool
    ):
        """Computes the next action, dispatching to training or eval action selection.

        Args:
            observation: Current observation from the environment.
            task_id: Current task id, typically an int.
            action_space_id: Current action space id.
            last_timestep_data: Data about the prior timestep (used to add the
                completed transition to the replay buffer during training).
            eval_mode: If True, computes an eval (greedy, no buffer update) action.

        Returns:
            Tuple `(action, timestep_data)`: the selected action, and a new
            `DQNTimestepData` recording this observation/action for use as
            `last_timestep_data` on the next call.

        Note:
            `continual_rl` calls this method for every environment step, both
            training and eval.
        """
        action_space = self.action_spaces[action_space_id]
        
        if eval_mode:
            # print(f"Eval Rand State: {np.random.get_state()[1][0]}")
            action = self._compute_eval_action(
                observation=observation,
                task_id=task_id, 
                action_space=action_space,
                last_timestep_data=last_timestep_data,
            )
        else:
            # print(f"Training Rand State: {np.random.get_state()[1][0]}")
            action = self._compute_training_action(
                observation=observation,
                task_id=task_id,
                action_space=action_space, 
                last_timestep_data=last_timestep_data
            ) 
            # Only iterate total_timesteps for training
            self.total_timesteps += self.n_envs

        # Continual RL handles the last_timestep_data so eval is not needed to be considered 
        timestep_data = DQNTimestepData(observation=observation, action=action)

        return action, timestep_data
    
    def _compute_training_action(
        self,
        observation: torch.Tensor, 
        task_id: Union[int, str], 
        action_space: Space, 
        last_timestep_data: DQNTimestepData,
    ) -> torch.Tensor:
        """Selects a training action (epsilon-greedy) and performs the surrounding bookkeeping.

        Args:
            observation: Current observation from the environment.
            task_id: Current task id, typically an int.
            action_space: Current action space.
            last_timestep_data: Data about the prior timestep, or None if a
                task just switched.

        Returns:
            The selected action tensor.

        Note:
            In order: 1) computes a greedy or random (epsilon-greedy) action;
            2) appends the last stored transition, if any, to the replay
            buffer; 3) updates the target DQN if it's due; 4) checks whether
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
            
        # Comes last so that the last_timestep_data sample corresponds to correct task_id
        # if last_timestep_data is used at all.
        self._update_task_info(task_id, action_space)
        
        return action
    
    def _compute_eval_action(        
        self,
        observation: torch.Tensor, 
        task_id: Union[int, str], 
        action_space: Space, 
        last_timestep_data: DQNTimestepData,
    ) -> torch.Tensor:
        """Selects a greedy (eval) action, temporarily switching the DQN into eval mode.

        Args:
            observation: Current observation from the environment.
            task_id: Current task id, typically an int. Unused, kept for
                signature parity with `_compute_training_action`.
            action_space: Current action space.
            last_timestep_data: Data about the prior timestep. Unused, kept for
                signature parity with `_compute_training_action`.

        Returns:
            The selected (greedy) action tensor.

        Note:
            Temporarily puts the DQN in eval mode (disabling dropout, batchnorm
            updates, etc.) for the forward pass, then restores training mode if
            it was previously training.
        """
        # If training, change to eval mode.
        training = self.dqn.training
        if training: self.dqn.eval()
        
        action = self._get_greedy_action(observation, action_space)
        
        # If training, revert back to training mode.
        if training: self.dqn.train()
        
        return action
    
    def _get_greedy_action(
        self, 
        observation: torch.Tensor, 
        action_space: Space
    ) -> torch.Tensor:
        """Selects the action with the highest predicted Q-value.

        Args:
            observation: Current observation from the environment.
            action_space: Current action space. Unused (the DQN's output
                dimensionality already matches the action space), kept for
                signature consistency.

        Returns:
            The greedy action tensor, on CPU.
        """
        _, q_values = self.dqn(observation.to(self.device))
        action = q_values.argmax(dim=1, keepdim=True).cpu()

        return action

    def _get_random_action(self,  action_space: Space):
        """Samples a uniformly random action from `action_space`, once per environment.

        Args:
            action_space: Current action space to sample from.

        Returns:
            A tensor of shape `(n_envs, 1)` containing random actions, dtyped
            to match the replay buffer's stored actions.
        """
        dtype = getattr(torch, str(self.replay_buffer.actions.dtype))
        return torch.tensor(
            [[action_space.sample()] for _ in range(self.n_envs)],
            dtype=dtype
        )

    def _update_task_info(
        self, 
        task_id: Union[int, str], 
        action_space: Space
    ) -> None:
        """Updates `self.task_id`/`self.current_action_size`, handling optimizer/buffer resets on task change.

        Args:
            task_id: Current task id, typically an int.
            action_space: Current action space.
        """
        # Set task_id and current action space
        if self.task_id is None:
            self.task_id = task_id
            self.current_action_size = action_space.n
            self.logger.info(f"Setting task - ID: {self.task_id} Action Size: {self.current_action_size}")
        # On task end, update task_id and action space when new task is seen
        elif self.task_id != task_id:
            if self.cfg.reset_optimizer:
                self.logger.info("Resetting optimizer(s) before new task.")
                self.init_model_optimizers()
            if self.cfg.reset_buffer:
                self.logger.info("Resetting replay buffer before new task.")
                self.replay_buffer.reset()
            self.current_action_size = action_space.n
            self.task_id = task_id
            self.logger.info(f"Setting task - ID: {self.task_id} Action Size: {self.current_action_size}")
    
    def train(self, storage_buffer):
        """Performs one gradient step on a minibatch sampled from the replay buffer, if enough data exists.

        Args:
            storage_buffer: Samples represented as a list of lists
                `[[DQNTimestepData, ...]]`. Each inner list represents the data
                collected by a single process since `train()` was last called.
                Unused here (data was already added to the replay buffer in
                `_compute_training_action`); generated by the `EnvironmentRunner`
                — see `EnvironmentRunnerBase.collect_data` for details.

        Returns:
            The Tensorboard-style log list from the loss computation, or None
            if there weren't yet enough samples in the replay buffer to train.

        Note:
            This method is only called every n steps, where n is
            `timesteps_per_collection`.
        """
        if not self.replay_buffer.full and self.cfg.update_after >= self.replay_buffer.pos:
            self.logger.debug(f"{self.total_timesteps} Only {len(self.replay_buffer)} samples, need {self.cfg.update_after} samples before training")
            return
        
        replay_samples = self.replay_buffer.sample(
            batch_size=self.cfg.batch_size,
            combined=self.cfg.combined_sampling,
        )
        # TODO: REMOVE debug statement 
        assert len(replay_samples) == self.cfg.batch_size

        loss, log = self._compute_loss(replay_samples)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return log

    def _compute_loss(self, replay_samples):
        """Computes the configured DQN loss (and its Tensorboard log entries) for a batch of samples.

        Args:
            replay_samples: A `Samples` batch drawn from the replay buffer.

        Returns:
            Tuple `(loss, log)`: the scalar loss tensor, and a list of
            Tensorboard-style log dicts (including the overall 'loss' scalar).
        """
        loss, loss_log = self.loss(
            dqn=self.dqn,
            target_dqn=self.target_dqn,
            replay_samples=replay_samples,
            **self.cfg.loss_kwargs
        )

        # Tensorboard tracking
        log = [
            {"type": "scalar", "tag": "loss", "value": loss},
        ] + loss_log
        
        return loss, log
        
    def save(self, output_path_dir, cycle_id, task_id, task_total_steps):
        """Saves the DQN/target DQN/optimizer state dicts, and optionally the replay buffer.

        Args:
            output_path_dir: Directory to save 'dqn.pt' (and, if configured,
                'replay_buffer.pkl') into.
            cycle_id: Current cycle. Unused, kept for `PolicyBase` API compatibility.
            task_id: Current task id. Unused, kept for `PolicyBase` API compatibility.
            task_total_steps: Current number of total steps. Unused, kept for
                `PolicyBase` API compatibility.

        Note:
            Loading still requires constructing the policy with its required
            arguments; `load` then overwrites the model and optimizer states in place.
        """
        self.logger.info(f"{self.total_timesteps} Saving policy: {output_path_dir}")
        checkpoint_data = {
            "dqn_state_dict": self.dqn.state_dict(),
            "target_dqn_state_dict": self.target_dqn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        model_path = os.path.join(output_path_dir, "dqn.pt")
        torch.save(checkpoint_data, model_path)
        
        if self.cfg.save_replay_buffer:
            replay_path = os.path.join(output_path_dir, 'replay_buffer.pkl')
            pickle_dump(self.replay_buffer, replay_path)
        
    def load(self, output_path_dir):
        """Loads the DQN/target DQN/optimizer state dicts, and the replay buffer if present.

        Args:
            output_path_dir: Directory containing a previously saved 'dqn.pt'
                (and optionally 'replay_buffer.pkl'), as written by `save`. If
                'dqn.pt' does not exist, model/optimizer state is left
                unchanged; likewise for the replay buffer.
        """
        model_path = os.path.join(output_path_dir, "dqn.pt")

        if os.path.exists(model_path):
            self.logger.info(f"Loading policy: {model_path}")
            checkpoint_data = torch.load(model_path)
            self.dqn.load_state_dict(checkpoint_data['dqn_state_dict'])
            self.target_dqn.load_state_dict(checkpoint_data['target_dqn_state_dict'])
            self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
        replay_path = os.path.join(output_path_dir, 'replay_buffer.pkl')
        if os.path.exists(replay_path):
            self.logger.info(f"Loading replay buffer: {replay_path}") 
            self.replay_buffer = pickle_load(replay_path)
            
