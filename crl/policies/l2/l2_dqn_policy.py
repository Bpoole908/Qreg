
from copy import deepcopy
from typing import List, Union
from pdb import set_trace

import torch
from gym import Space

from crl.policies.dqn.dqn_policy import DQNPolicy
from crl.policies.l2.l2_dqn_policy_config import L2DQNPolicyConfig


class L2DQNPolicy(DQNPolicy):
    """DQN with L2 weight regularization against forgetting (encoder weights only).

    Penalizes changes to the DQN encoder's weights relative to their value at
    the end of the previous task, scaled by a growing `importance` factor as
    more tasks are seen. Only the most recently completed task's weights are
    tracked as the regularization target (not one snapshot per task).

    Reference:
        COOM: https://github.com/hyintell/COOM/blob/e77b1ce40503ae73e46e73e2506aa80cfe30ecc9/CL/methods/l2.py#L7
    """

    def __init__(
        self,
        config: L2DQNPolicyConfig,
        observation_space: List[Space],
        action_spaces: List[Space]
    ):
        """
        Args:
            config: Policy configuration, including the L2-specific
                `reg_coef` hyperparameter.
            observation_space: Gym observation space, shared across tasks.
            action_spaces: Dict mapping action_space_id to a Gym action space;
                all must have the same `.shape`.
        """
        super().__init__(config, observation_space, action_spaces)
        self.reg_task_weights = {}
        self.importance = 0

    def _update_task_info(
        self,
        task_id: Union[int, str],
        action_space: Space
    ) -> None:
        """Snapshots regularization weights and bumps `importance` on task change, then defers to `DQNPolicy`.

        Args:
            task_id: Current task id, typically an int.
            action_space: Current action space.
        """
        if self.task_id is not None and self.task_id != task_id:
            self.logger.info(f"Storing weights for task ID {self.task_id}")
            self.update_reg_task_weights()
            # Scales regularization as more tasks are seen, mimics COOM implementation
            self.importance += 1 
            self.logger.info(f"Updating importance value to {self.importance}")

        super()._update_task_info(task_id, action_space)

    def update_reg_task_weights(self):
        """Snapshots the current DQN encoder's trainable weights as the L2 regularization target."""
        for name, param in self.dqn.encoder.named_parameters():
            if param.requires_grad:
                self.reg_task_weights[name] = param.data.clone()
        self.logger.info(f"Updated reg task weights for {len(self.reg_task_weights)} weights/biases")

    def _compute_loss(self, replay_samples):
        """Computes the DQN replay loss plus the L2 regularization penalty.

        Args:
            replay_samples: A `Samples` batch drawn from the replay buffer.

        Returns:
            Tuple `(loss, log)`: the combined loss (replay + L2 penalty), and
            a list of Tensorboard-style log dicts including the total,
            replay-only, and L2-only loss components.
        """
        l2_loss, l2_log = 0, []
        # Replay Loss
        replay_loss, loss_log = self.loss(
            dqn=self.dqn,
            target_dqn=self.target_dqn,
            replay_samples=replay_samples,
            **self.cfg.loss_kwargs
        )
        # L2 Loss
        if len(self.reg_task_weights) != 0:
            l2_loss, l2_log = self._l2_loss()

        loss = replay_loss + l2_loss  
        log = [
            {"type": "scalar", "tag": "loss", "value": loss},
            {"type": "scalar", "tag": "replay_loss", "value": replay_loss},
            {"type": "scalar", "tag": "l2_loss", "value": l2_loss},
        ] + loss_log + l2_log
        
        return loss, log
    
    def _l2_loss(self):
        """Computes the importance-weighted L2 penalty between current and stored encoder weights.

        Returns:
            Tuple `(loss, log)`: the scaled (by `cfg.reg_coef`) L2 penalty, and
            an empty log list.
        """
        loss, log = 0, []
        for name, param  in self.dqn.encoder.named_parameters():
            if name in self.reg_task_weights:
                task_param = self.reg_task_weights[name]
                loss += torch.sum(self.importance * (param - task_param)**2)
                
        return self.cfg.reg_coef * loss, log