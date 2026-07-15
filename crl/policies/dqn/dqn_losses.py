from pdb import set_trace

import numpy as np
import torch
from torch.nn import functional as F    


class DQNLosses():
    """Namespace of static DQN temporal-difference loss functions, selected by name via `DQNPolicy.get_loss`."""

    @staticmethod
    def dqn(
        dqn,
        target_dqn,
        replay_samples,
        *,
        discount_factor = .99,
        reward_clipping_func = None,
    ):
        """Computes the basic DQN Huber loss between predicted and target Q-values.

        The target Q-value bootstraps off the target network's max next-state
        Q-value. Keyword-only args (after `*`) are configured via
        `DQNPolicyConfig.loss_kwargs`, not passed positionally by the policy.

        Args:
            dqn: The online `DQN` model, used to compute current Q-values.
            target_dqn: The target `DQN` model, used to compute bootstrapped
                next-state Q-values.
            replay_samples: A `Samples` batch with observations, actions,
                rewards, dones, and next_observations.
            discount_factor: Discount value for future rewards.
            reward_clipping_func: Optional function applied to rewards before
                computing the loss (e.g. clipping to [-1, 1]).

        Returns:
            Tuple `(loss, log)`: the scalar Huber loss, and a list of
            Tensorboard-style log dicts with per-action mean Q-values.
        """
        with torch.no_grad():
            _, target_next_q_values = target_dqn(replay_samples.next_observations)
            # Select only the viable actions
            next_q_value, _ = target_next_q_values.max(dim=1, keepdim=True)
            
            if reward_clipping_func is not None:
                rewards = reward_clipping_func(replay_samples.rewards)
            else:
                rewards = replay_samples.rewards

            target_q_value = (
                rewards
                + (1. - replay_samples.dones) 
                * discount_factor
                * next_q_value
            )
            
        _, q_values = dqn(replay_samples.observations)
        # Get all q values corresponding to the selected actions
        q_value = torch.gather(
            q_values, 
            dim=1, 
            index=replay_samples.actions
        )
        
        q_value_means = torch.mean(q_values, axis=0)
        log = [{"type": "scalar", "tag": f"q_value_{c}", "value": m} for c, m in enumerate(q_value_means)]
        # Reduction is mean by default
        return F.huber_loss(q_value, target_q_value), log
    
    @staticmethod
    def double_dqn(
        dqn,
        target_dqn,
        replay_samples,
        *,
        discount_factor = .99,
        reward_clipping_func = None,
    ):
        """Computes the Double DQN Huber loss between predicted and target Q-values.

        Unlike plain DQN, the next action is selected using the online network
        and evaluated using the target network, reducing overestimation bias.
        Keyword-only args (after `*`) are configured via
        `DQNPolicyConfig.loss_kwargs`, not passed positionally by the policy.

        Args:
            dqn: The online `DQN` model, used to compute current Q-values and
                to select the next action.
            target_dqn: The target `DQN` model, used to evaluate the
                online-selected next action's Q-value.
            replay_samples: A `Samples` batch with observations, actions,
                rewards, dones, and next_observations.
            discount_factor: Discount value for future rewards.
            reward_clipping_func: Optional function applied to rewards before
                computing the loss (e.g. clipping to [-1, 1]).

        Returns:
            Tuple `(loss, log)`: the scalar Huber loss, and a list of
            Tensorboard-style log dicts with per-action mean Q-values.
        """
        with torch.no_grad():
            _, target_next_q_values = target_dqn(replay_samples.next_observations)
            _, next_q_values = dqn(replay_samples.next_observations)
            
            next_action = next_q_values.argmax(dim=1, keepdim=True)
            # Get all the indexes corresponding to the next_action
            next_q_value = torch.gather(
                target_next_q_values, 
                dim=1, 
                index=next_action
            )

            if reward_clipping_func is not None:
                rewards = reward_clipping_func(replay_samples.rewards)
            else:
                rewards = replay_samples.rewards

            target_q_value = (
                rewards
                + (1. - replay_samples.dones) 
                * discount_factor
                * next_q_value
            )
        
        _, q_values = dqn(replay_samples.observations)
        # Get all q values corresponding to the selected actions
        q_value = torch.gather(
            q_values, 
            dim=1, 
            index=replay_samples.actions
        )
        
        q_value_means = torch.mean(q_values, axis=0)
        log = [{"type": "scalar", "tag": f"q_value_{c}", "value": m} for c, m in enumerate(q_value_means)]
        
        return F.huber_loss(q_value, target_q_value), log
            
