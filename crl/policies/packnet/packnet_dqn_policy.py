import warnings
from typing import Union
from pdb import set_trace
from copy import deepcopy

import torch
from tqdm.auto import tqdm
import numpy as np
import gym
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from crl.policies.dqn.dqn_policy import DQNPolicy
from crl.policies.dqn.dqn_timestep_data import DQNTimestepData
from crl.policies.packnet.packnet_dqn_policy_config import PackNetDQNPolicyConfig
from crl.policies.packnet.model import PackNetDQN

class SparsePruner():
    """Performs pruning on a PyTorch model for Conv2D and Linear layers. 
    
        Attributes:
            model: Pytorch model to be pruned.
            
            starting_seq_idx: The starting sequence index number. This will be iterated by
                1 after each pruning. The sequence index will represent the current task
                or dataset being used for training. 
                
            train_bias: Determines whether or not bias gradients will be set to zero
                or not.
            
            train_norm: Determines whether or not layer/batch norm gradients will be set to zero
                or not.
                
            module_data: Contains masks for each layer and corresponding module reference.
                Each mask will set the masked weights equal to the seq_idx given.
                
            pruned_idx: Used to mark weights that have been pruned and will be used for
                training in the next task. After each pruning the pruned_idx is iterated
                by 1 and assumes the next seq_idx values will be seq_idx + 1.
    """

    def __init__(
        self,
        model,
        starting_seq_idx: int = 0,
        train_bias: bool = True,
        train_norm: bool = True
    ):
        """
        Args:
            model: PyTorch model whose Conv2d/Linear layers will be pruned.
            starting_seq_idx: Initial sequence index (task index) that all
                weights are assigned to before any pruning has occurred.
            train_bias: If False, bias gradients are zeroed (frozen) in
                `make_grads_zero`.
            train_norm: If False, layer/batch-norm gradients are zeroed
                (frozen) in `make_grads_zero`.
        """
        self.model = model
        self.train_bias = train_bias
        self.train_norm = train_norm

        self.module_data = {}
        # Extract all relevant layers that can be pruned (e.g., Conv2d and linear)
        for module_idx, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = torch.Tensor(module.weight.data.size()).fill_(starting_seq_idx)
                if 'cuda' in module.weight.data.type():
                    mask = mask.cuda()
                self.module_data[module_idx] = {'mask': mask, 'module': module}
        # Set the pruned index to the expected sequence index for the next task.
        self.pruned_idx = starting_seq_idx + 1
    
    def pruning_mask(self, weights, mask, layer_idx, prune_perc, seq_idx):
        """Computes an updated mask that marks the smallest-magnitude `prune_perc` fraction of weights owned by `seq_idx` as pruned.

        Args:
            weights: The layer's weight tensor.
            mask: Current mask tensor (same shape as `weights`), where each
                entry holds the sequence index that "owns" that weight.
            layer_idx: Index of this layer, used only for logging.
            prune_perc: Fraction (0-1) of `seq_idx`'s currently-owned weights
                to mark as pruned.
            seq_idx: The sequence index (task) whose weights are being pruned.

        Returns:
            The updated mask, with the smallest-magnitude `prune_perc` of
            `seq_idx`'s weights reassigned to `self.pruned_idx`.

        Raises:
            AssertionError: If `seq_idx` is not present in `mask`.
        """
        assert seq_idx in mask, f"Can not prune seq_idx {seq_idx} as it is not in the mask."
        # Select all prunable weights, ie. belonging to current dataset.
        mask = mask.cuda()
        tensor = weights[mask.eq(seq_idx)]
        abs_tensor = tensor.abs()
        cutoff_rank = round(prune_perc * tensor.numel())
        cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0].item()
        # Remove those weights which are below cutoff and belong to current
        # dataset that we are training for.
        # WARNING: If mask.eq( self.pruned_idx ).sum() is greater than the cutoff_rank 
        #          then all prior seq_idx could get overridden. This can be
        #           due to ReLU or any other activation that can saturate towards 0.
        #           Thus the kthvalue is equal to the min and more weights can get pruned 
        #           than intended.  
        remove_mask = weights.abs().le(cutoff_value) * mask.eq(seq_idx)
        if remove_mask.sum() > cutoff_rank:
            diff = (remove_mask.sum() - cutoff_rank).item()
            msg = f"remove_mask {remove_mask.sum()} is larger than estimated pruning {cutoff_rank}. " \
                  f"Attempting to automatically adjust by setting {diff} random remove_mask values to False."
            warnings.warn(msg)
            locs = torch.where(remove_mask == True)
            loc_idx = np.random.choice(np.arange(len(locs[0])), size=diff, replace=False)
            new_locs = []
            for l in locs:
                new_locs.append(l[loc_idx])
            remove_mask[new_locs] = False
        assert remove_mask.sum() == cutoff_rank
        mask[remove_mask.eq(1)] = self.pruned_idx 

        print(f"Rank: {cutoff_rank} Value: {cutoff_value}")
        print('Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_idx, mask.eq( self.pruned_idx ).sum(), tensor.numel(),
               100 * mask.eq( self.pruned_idx ).sum() / tensor.numel(), weights.numel()))

        return mask

    def prune(self, prune_perc: float, seq_idx: int, pruned_idx: int = None):
        """Prunes `prune_perc` of `seq_idx`'s weights in every tracked layer, zeroing them out.

        Updates each layer's mask via `pruning_mask` and zeroes the
        corresponding weight values, then advances `self.pruned_idx` for the next task.

        Args:
            prune_perc: Fraction (0-1) of `seq_idx`'s weights to prune. Must be nonzero.
            seq_idx: The sequence index (task) whose weights are being pruned.
            pruned_idx: If given, sets `self.pruned_idx` to this value instead
                of incrementing it by 1.

        Raises:
            AssertionError: If `prune_perc` is 0.
        """
        assert prune_perc != 0

        for module_idx, data in self.module_data.items():
            module = data['module']
            
            data['mask'] = self.pruning_mask(
                module.weight.data, 
                data['mask'], 
                module_idx, # TODO: remove
                prune_perc,
                seq_idx
            )
            # Set pruned weights to 0.
            module.weight.data[data['mask'].eq(self.pruned_idx)] = 0.0

        # Set the next pruned index identifier.
        if pruned_idx is not None:
            self.pruned_idx = pruned_idx
        else:
            self.pruned_idx += 1
        
    def make_grads_zero(self, seq_idx: int):
        """Zeroes gradients for weights not owned by `seq_idx`, and (if configured) frozen bias/norm params.

        Called before the optimizer step so that only the current task's
        allotted weights are updated.

        Args:
            seq_idx: The current task's sequence index; weights owned by any
                other sequence index have their gradients zeroed.

        Raises:
            AssertionError: If no prunable layers were found (`module_data` is empty).
        """
        assert len(self.module_data) > 0
        # Loop over all modules
        for module_idx, module in enumerate(self.model.modules()):
            # Only mask known modules
            if module_idx in self.module_data:
                layer_mask = self.module_data[module_idx]['mask']
                assert seq_idx in layer_mask
                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[layer_mask.ne(seq_idx)] = 0
                    if not self.train_bias:
                        # Biases are fixed.
                        if module.bias is not None:
                            module.bias.grad.data.fill_(0)
            # Check for any norm layers that might need to have grads zeroed out
            elif isinstance(module, (nn.LayerNorm, _BatchNorm)):
                # Set grads of norm params to 0.
                if not self.train_norm:
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)

    def set_view(self, seq_idx: int) -> None:
        """Bring back the version of the models from a moment corresponding to a given task.

        Args:
            seq_idx: Number of a task. If this value is N >= 0, then the weights corresponding to
            tasks 0, 1, ..., N will be set to their real values, and the rest will be set to 0. This
            is used to do inference on task N.
            If seq_idx is -1, all weights will be set to their current values. This mode is used in
            training.
        """
        if seq_idx == -1:
            for module_idx, data in self.module_data.items():
                data['module'].weight.data = data['saved']
        else:
            for module_idx, data in self.module_data.items():
                data['saved'] = data['module'].weight.data.clone()
                data['module'].weight.data = data['module'].weight.data * (data['mask'] <= seq_idx)
            # print(f"SET VIEW {seq_idx} -- UNIQUES: {torch.unique(data['mask'][data['mask'] <= seq_idx]) }")

    
class PackNetDQNPolicy(DQNPolicy):
    """DQN with PackNet: after each task, prunes and retrains to free up capacity for the next task.

    Implements the parameter-isolation continual-learning method PackNet
    (adapted per COOM): each task is allotted a shrinking
    fraction of the network's remaining free weights (pruned via
    `SparsePruner`), and gradients are masked during training so a task only
    ever updates its own allotted weights. During the first cycle through all
    tasks, evaluation on an earlier task restricts the visible weights to
    those allotted up through that task (`_compute_eval_action`); after the
    first cycle, all weights are considered "seen together" and this
    restriction is no longer needed.
    """

    def __init__(self, config: PackNetDQNPolicyConfig, observation_space, action_spaces):
        """
        Args:
            config: Policy configuration, including PackNet-specific
                hyperparameters (`retrain_steps`, `num_tasks`).
            observation_space: Gym observation space, shared across tasks.
            action_spaces: Dict mapping action_space_id to a Gym action space;
                all must have the same `.shape`.

        Raises:
            AssertionError: If `config.num_tasks` is not an int.
        """
        super().__init__(config, observation_space, action_spaces)
        self.total_tasks = 0
        self.pruner = SparsePruner(self.dqn)
        self._task2seq = {}
        assert isinstance(self.cfg.num_tasks, int)

    @property
    def on_first_cycle(self):
        """bool: True until every task has been seen once (`total_tasks < cfg.num_tasks`)."""
        return self.total_tasks < self.cfg.num_tasks

    def init_models(self):
        """Builds the online `PackNetDQN` and target `PackNetDQN` networks and syncs the target to the online weights."""
        n_actions = self.action_space.n
        self.dqn = PackNetDQN(
            observation_space=self.observation_space, 
            output_dims=n_actions,
            **self.cfg.dqn_kwargs
        ).to(self.device)
        self.target_dqn = PackNetDQN(
            observation_space=self.observation_space, 
            output_dims=n_actions,
            **self.cfg.dqn_kwargs
        ).to(self.device)
        self._update_target_model()

    def seq_idx(self, task_id: Union[int, str] = None):
        """Gets the sequence index (weight-ownership index) assigned to a task.

        Args:
            task_id: The task ID to look up. If None, uses `self.task_id`.

        Returns:
            The sequence index previously assigned to `task_id` in `self._task2seq`.
        """
        return self._task2seq[task_id if task_id is not None else self.task_id]

    def _update_task_info(self, task_id, action_space):
        """Sets up the first task, or triggers `on_task_end` (pruning/retraining) when a new task begins.

        Args:
            task_id: Current task id, typically an int.
            action_space: Current action space.
        """
        if self.task_id is None:
            self.task_id = task_id
            self.current_action_size = action_space.n
            if task_id not in self._task2seq:
                self._task2seq[task_id] = self.total_tasks
            self.logger.info(f"Setting task - ID: {self.task_id} Seq Idx: {self.seq_idx()} Action Size: {self.current_action_size}")
        # Update task_id and action space when new task is seen
        elif self.task_id != task_id:
            self.on_task_end(task_id, action_space)

    def on_task_end(self, task_id: Union[int, str], action_space: gym.spaces.Space):
        """Prunes the just-finished task's weights (if still in the first cycle), retrains, and switches to the new task.

        Only prunes/retrains during the first cycle through all tasks
        (`on_first_cycle`); in later cycles, all weights have already been
        allotted so there's nothing left to prune. Prune percentage is chosen
        so that each remaining task gets an equal share of the currently-owned
        weights. After pruning, optionally resets the optimizer and retrains
        for `cfg.retrain_steps` steps so the surviving weights adapt to
        losing the pruned ones. Finally updates task/optimizer/buffer state
        for the new task, assigning it a fresh sequence index if unseen.

        Args:
            task_id: The new task id (the just-ended task is `self.task_id`,
                which will differ from `task_id`).
            action_space: The action space corresponding to `task_id`.
        """
        if self.on_first_cycle:
            # TODO: Debug REMOVE
            for module_idx, data in self.pruner.module_data.items():
                seq_ids, counts = torch.unique(data['mask'], return_counts=True)
                self.logger.debug(f"Module {module_idx}: Seq IDs: {seq_ids} Counts: {counts}")
            # Compute percent of weights to prune so same for each task
            num_tasks_left = self.cfg.num_tasks - self.seq_idx() - 1
            prune_perc = num_tasks_left / (num_tasks_left + 1)
            
            if prune_perc != 0:
                # Prune weights
                self.logger.info(f"Pruning for task ID {self.task_id} with seq ID {self.seq_idx()} by {100 * prune_perc:.2f}%")
                self.pruner.prune(prune_perc, self.seq_idx())
                
                # Reset optimizer before retraining
                if self.cfg.reset_optimizer:
                    self.logger.info("Resetting optimizer(s) before retraining.")
                    self.init_model_optimizers() 
                       
                # Retrain after pruning weights
                self.logger.info(f"Retraining task ID {self.task_id} with seq ID {self.seq_idx()} for {self.cfg.retrain_steps} steps.")
                for _ in tqdm(range(self.cfg.retrain_steps)):
                    self.train()
        
        # TODO: Debug REMOVE
        for module_idx, data in self.pruner.module_data.items():
            seq_ids, counts = torch.unique(data['mask'], return_counts=True)
            self.logger.debug(f"Module {module_idx}: Seq IDs: {seq_ids} Counts: {counts}")
        
        # NOTE: COOM will freeze bias and norm values after the first task.
        if self.cfg.reset_optimizer:
            self.logger.info("Resetting optimizer(s) before new task.")
            self.init_model_optimizers()
        if self.cfg.reset_buffer:
            self.logger.info("Resetting replay buffer before new task.")
            self.replay_buffer.reset()
        self.current_action_size = action_space.n
        self.task_id = task_id
        self.total_tasks += 1
        # Set the seq index for the new task ID
        if task_id not in self._task2seq:
            self._task2seq[task_id] = self.total_tasks
        self.logger.info(f"Setting task - ID: {self.task_id} Seq Idx: {self.seq_idx()} Action Size: {self.current_action_size}")
            
    def train(self, storage_buffer = None):
        """Performs one gradient step, masking gradients so only the current task's owned weights update.

        Args:
            storage_buffer: Unused; accepted for `DQNPolicy.train` API
                compatibility (also called directly, with no argument, during
                post-pruning retraining in `on_task_end`).

        Returns:
            The Tensorboard-style log list from the loss computation, or None
            if there weren't yet enough samples in the replay buffer to train.

        Note:
            `continual_rl` calls this method after a specified number of steps
            (`timesteps_per_collection`).
        """
        if not self.replay_buffer.full and self.cfg.update_after >= self.replay_buffer.pos:
            self.logger.debug(f"{self.total_timesteps} Only {len(self.replay_buffer)} samples, need {self.cfg.update_after} samples before training")
            return
        
        replay_samples = self.replay_buffer.sample(
            batch_size=self.cfg.batch_size,
            combined=self.cfg.combined_sampling,
        )
        
        loss, logs = self._compute_loss(replay_samples)
        self.optimizer.zero_grad()
        loss.backward()
        self.pruner.make_grads_zero(self.seq_idx())
        self.optimizer.step()
        
        return logs
    
    def _compute_eval_action(
        self, 
        observation: np.ndarray, 
        task_id: Union[int, str], 
        action_space: gym.spaces.Space, 
        last_timestep_data: DQNTimestepData
    ):
        """Selects a greedy eval action, temporarily restricting visible weights to those allotted through `task_id`.

        During the first cycle through all tasks, restricts the weight view to
        sequence indices `0..seq_idx(task_id)` before acting (e.g. if training
        task 5 but evaluating task 3, only weights allotted to tasks 0-3 are
        used, since task 3 hasn't yet been trained alongside task 5's
        weights). After the first cycle, no restriction is needed since every
        task has already been trained together.

        Args:
            observation: Current observation from the environment.
            task_id: Current task ID.
            action_space: Current action space.
            last_timestep_data: Data about the prior timestep. Unused, kept
                for signature parity with `DQNPolicy._compute_eval_action`.

        Returns:
            The selected (greedy) action tensor.

        Note:
            Sets the weight view for every action, which is not ideal if the
            view isn't changing often. Additionally, eval-only tasks always
            use ALL weights, since they are never added to `self._task2seq`.
        """
        # Set weights to be weights up to and including the passed task ID.
        # This is until you reach the 2nd cycle when all task/weights have been seen together.
        if self.on_first_cycle and task_id in self._task2seq:
            self.pruner.set_view(self.seq_idx(task_id))
        action = self._get_greedy_action(observation, action_space)
        # Set view back to all weights
        if self.on_first_cycle and task_id in self._task2seq:
            self.pruner.set_view(-1)

        return action
        