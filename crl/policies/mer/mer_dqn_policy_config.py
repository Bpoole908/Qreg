from crl.policies.dqn.dqn_policy_config import DQNPolicyConfig, get_tag_value
from crl.common.yaml import convert_float_to_str as f2s


def experiment_tag(policy, exp):
    """Builds a short experiment tag string summarizing a MER-DQN policy's non-default hyperparameters.

    Args:
        policy: The policy instance, whose `policy_kwargs` dict is inspected.
       exp: The raw experiment config.

    Returns:
        An underscore-joined tag string, including only the hyperparameters
        that differ from `MERDQNPolicyConfig` defaults or are otherwise notable.
    """
    defaults = MERDQNPolicyConfig()
    pk = policy.policy_kwargs
    tag = [
        get_tag_value(pk, 'buffer_size', 'rb={}', '', transform_value=f2s),
        get_tag_value(pk, 'learning_rate', 'lr={}', ''),
        get_tag_value(pk, 'timesteps_per_collection', 'tpc={}', '', 
                      check_value=lambda x: x if x != defaults.timesteps_per_collection else False),
        get_tag_value(pk, 'update_after', 'upaf={}', '', check_value=lambda x: x, transform_value=f2s),
        get_tag_value(pk, 'within_batch_beta', 'beta={}', ''),
        get_tag_value(pk, 'across_batch_gamma', 'gamma={}', ''),
        get_tag_value(pk, 'steps', 'steps={}', ''),
        get_tag_value(pk, 'online', 'batch','', check_value=lambda x: True if x == False else False),
    ]

    # Reset tags
    tag.extend([
        get_tag_value(pk, 'reset_buffer', 'rsrb','', check_value=lambda x: x if x == True else False),
        get_tag_value(pk, 'reset_optimizer', 'rsopt', '', check_value=lambda x: x if x == True else False),
    ])
        
    return '_'.join(filter(None, tag)) 


class MERDQNPolicyConfig(DQNPolicyConfig):
    """Hyperparameter/config container for `MERDQNPolicy`, extending `DQNPolicyConfig`.

    Attributes:
        across_batch_gamma: Reptile interpolation factor for the outer
            across-batch update.
        within_batch_beta: Reptile interpolation factor for the inner
            within-batch update.
        steps: Number of times to sample from the replay buffer (inner
            iterations) per `train()` call.
        online: If True, computes the loss/gradient per-sample within each
            sampled batch (true online MER). If False, uses full-minibatch
            learning instead.
    """

    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.buffer_size = 50000
        self.learning_rate = 0.0001
        self.combined_sampling = True
        
        self.within_batch_beta = 1
        self.across_batch_gamma = 0.3
        self.steps = 1
        self.online = True
