
from copy import deepcopy
from typing import List, Union
from pdb import set_trace

import torch
from gym import Space

from crl.policies.dqn.dqn_policy import DQNPolicy
from crl.policies.ewc.ewc_dqn_policy_config import EWCDQNPolicyConfig


class EWCDQNPolicy(DQNPolicy):
    """DQN with online Elastic Weight Consolidation (EWC) regularization against forgetting.

    Penalizes changes to weights that were important (per an online-updated
    Fisher information matrix) for previously seen tasks. Only the most
    recently completed task's weights are tracked as the regularization target
    (not one snapshot per task).

    References:
        CORA: https://github.com/AGI-Labs/continual_rl/tree/f2754bb282757829765beb4703f24b87efa13ff9/continual_rl/policies/ewc
        COOM: https://github.com/hyintell/COOM/blob/e77b1ce40503ae73e46e73e2506aa80cfe30ecc9/CL/methods/ewc.py#L8
    """

    def __init__(
        self,
        config: EWCDQNPolicyConfig,
        observation_space: List[Space],
        action_spaces: List[Space]
    ):
        """
        Args:
            config: Policy configuration, including EWC-specific
                hyperparameters (`fisher_lambda`, `fisher_num_batches`, `fisher_gamma`).
            observation_space: Gym observation space, shared across tasks.
            action_spaces: Dict mapping action_space_id to a Gym action space;
                all must have the same `.shape`.
        """
        super().__init__(config, observation_space, action_spaces)
        self.reg_task_weights = {}
        self.fishers = {name: torch.zeros(param.shape, device=self.device) 
                        for name, param in self.dqn.named_parameters()}
        self.fisher_norm = {name: 1 for name, _ in self.dqn.named_parameters()}

    def _update_task_info(
        self, 
        task_id: Union[int, str], 
        action_space: Space
    ) -> None:
        """Updates the Fisher matrix/regularization weights on task change, then defers to `DQNPolicy`.

        Args:
            task_id: Current task id, typically an int.
            action_space: Current action space.

        Note:
            Called before `super()._update_task_info`, so EWC regularization is
            never applied against the very first task (there's nothing to
            regularize towards yet).
        """
        # Since we call this before super, EWC reg will never be applied to the first task
        if self.task_id is not None and self.task_id != task_id:
            self.logger.info(f"Storing weights for task ID {self.task_id}")
            self.update_fisher_matrix()
            self.update_reg_task_weights()

        super()._update_task_info(task_id, action_space)

    def update_reg_task_weights(self):
        """Snapshots the current DQN's trainable weights as the EWC regularization target."""
        for name, param in self.dqn.named_parameters():
            if param.requires_grad:
                self.reg_task_weights[name] = param.data.clone()
        self.logger.info(f"Updated reg task weights for {len(self.reg_task_weights)} weights/biases")

    def update_fisher_matrix(self):
        """Estimates the diagonal Fisher information matrix from replay-buffer samples and folds it into the running (online) Fisher estimate.

        Averages squared gradients of the DQN loss over `cfg.fisher_num_batches`
        sampled batches to estimate the current Fisher matrix, then combines it
        with the previous online Fisher estimate (decayed by `cfg.fisher_gamma`)
        and renormalizes, per the CORA online-EWC update.
        """
        fishers = {}
        for _ in range(self.cfg.fisher_num_batches):
            replay_samples = self.replay_buffer.sample(
            batch_size=self.cfg.batch_size,
            combined=self.cfg.combined_sampling,
            )
            loss, _ = self.loss(
                dqn=self.dqn,
                target_dqn=self.target_dqn,
                replay_samples=replay_samples,
                **self.cfg.loss_kwargs
            )
            self.optimizer.zero_grad()
            loss.backward()
            for name, param in self.dqn.named_parameters():
                if param.requires_grad:
                    fishers[name] = fishers.get(name, 0) + param.grad**2 # torch.clip(fisher, 1e-5, torch.inf)
        self.optimizer.zero_grad()

        # Normalize over batches
        fishers = {n: p / self.cfg.fisher_num_batches for n, p in fishers.items()}

        # Online EWC update
        for name, old_fisher in self.fishers.items():
            new_fisher = fishers[name]
            new_fisher_norm = new_fisher / self.fisher_norm[name] 
            # NOTE: Normalize new_fisher using the prior norm as new_fisher can be extremely small value
            fisher = (self.cfg.fisher_gamma * old_fisher) + new_fisher_norm
            # Normalize online fisher matrix, taken from CORA 
            # https://github.com/AGI-Labs/continual_rl/blob/f2754bb282757829765beb4703f24b87efa13ff9/continual_rl/policies/ewc/ewc_monobeast.py#L277
            self.fisher_norm[name] = torch.norm(fisher)
            self.fishers[name] = fisher / self.fisher_norm[name] 
            
            self.logger.debug(
                f'{"{:=^50}".format(name)}\n' \
                f"New: {new_fisher.mean().item()}\n" \
                f"Normed New: {new_fisher_norm.mean().item()}\n" \
                f"Fisher: {fisher.mean().item()}\n" \
                f"Normed Fisher: {self.fishers[name].mean().item()}\n" \
                f"Fisher Norm: {self.fisher_norm[name]}"
            )
            
    def _compute_loss(self, replay_samples):
        """Computes the DQN replay loss plus the EWC regularization penalty.

        Args:
            replay_samples: A `Samples` batch drawn from the replay buffer.

        Returns:
            Tuple `(loss, log)`: the combined loss (replay + EWC penalty), and
            a list of Tensorboard-style log dicts including the total,
            replay-only, and EWC-only loss components.
        """
        ewc_loss, ewc_log = 0, []
        # Replay Loss
        replay_loss, loss_log = self.loss(
            dqn=self.dqn,
            target_dqn=self.target_dqn,
            replay_samples=replay_samples,
            **self.cfg.loss_kwargs
        )
        # EWC Loss
        if len(self.reg_task_weights) != 0:
            ewc_loss, ewc_log = self._ewc_loss()
            
        loss = replay_loss + ewc_loss  
        log = [
            {"type": "scalar", "tag": "loss", "value": loss},
            {"type": "scalar", "tag": "replay_loss", "value": replay_loss},
            {"type": "scalar", "tag": "ewc_loss", "value": ewc_loss},
        ] + loss_log + ewc_log
        
        return loss, log
    
    def _ewc_loss(self):
        """Computes the Fisher-weighted L2 penalty between current and stored regularization weights.

        Returns:
            Tuple `(loss, log)`: the scaled (by `cfg.fisher_lambda`) EWC penalty,
            and an empty log list.
        """
        loss, log = 0, []
        for name, param in self.dqn.named_parameters():
            if name in self.reg_task_weights:
                fisher = self.fishers[name]
                task_param = self.reg_task_weights[name]
                loss += torch.sum(fisher * (param - task_param)**2)
        # print(reg_loss.item(), (self.cfg.fisher_lambda * loss).item())
        loss = self.cfg.fisher_lambda * loss

        return loss, log