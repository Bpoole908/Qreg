from crl.policies.dqn.dqn_policy_config import DQNPolicyConfig, get_tag_value
from crl.common.yaml import convert_float_to_str as f2s


def experiment_tag(policy, exp):
    """Builds a short experiment tag string summarizing an EWC-DQN policy's non-default hyperparameters.

    Args:
        policy: The policy instance, whose `policy_kwargs` dict is inspected.
        exp: The raw experiment config.

    Returns:
        An underscore-joined tag string, including only the hyperparameters
        that differ from `EWCDQNPolicyConfig` defaults or are otherwise notable.
    """
    defaults = EWCDQNPolicyConfig()
    pk = policy.policy_kwargs
    tag = [
        get_tag_value(pk, 'buffer_size', 'rb={}', '', transform_value=f2s),
        get_tag_value(pk, 'learning_rate', 'lr={}', ''),
        get_tag_value(pk, 'timesteps_per_collection', 'tpc={}', '', 
                      check_value=lambda x: x if x != defaults.timesteps_per_collection else False),
        get_tag_value(pk, 'update_after', 'upaf={}', '', check_value=lambda x: x, transform_value=f2s),
        get_tag_value(pk, 'fisher_num_batches', 'nbatches={}', '', transform_value=f2s , 
                      check_value=lambda x: x != defaults.fisher_num_batches),
        get_tag_value(pk, 'fisher_lambda', 'l={}', '', transform_value=f2s),
        get_tag_value(pk, 'fisher_gamma', 'g={}', '', transform_value=f2s),
    ]
    
    # Reset tags
    tag.extend([
        get_tag_value(pk, 'reset_buffer', 'rsrb','', check_value=lambda x: x if x == True else False),
        get_tag_value(pk, 'reset_optimizer', 'rsopt', '', check_value=lambda x: x if x == True else False),
    ])
        
    return '_'.join(filter(None, tag)) 


class EWCDQNPolicyConfig(DQNPolicyConfig):
    """Hyperparameter/config container for `EWCDQNPolicy`, extending `DQNPolicyConfig`.

    Attributes:
        fisher_lambda: Scaling for the EWC (Fisher-weighted L2) regularization term.
        fisher_num_batches: Number of batches used to estimate the Fisher
            matrix. Uses the DQN `batch_size` as the per-batch size.
        fisher_gamma: Online-EWC decay term determining how much the Fisher
            matrix estimate forgets prior tasks when updated.
    """

    def __init__(self):
        super().__init__()
        self.fisher_lambda = 10000
        self.fisher_num_batches = 100
        self.fisher_gamma = 0.95
