from crl.policies.dqn.dqn_policy_config import DQNPolicyConfig, get_tag_value
from crl.common.yaml import convert_float_to_str as f2s


def experiment_tag(policy, exp):
    """Builds a short experiment tag string summarizing an L2-DQN policy's non-default hyperparameters.

    Args:
        policy: The policy instance, whose `policy_kwargs` dict is inspected.
       exp: The raw experiment config.

    Returns:
        An underscore-joined tag string, including only the hyperparameters
        that differ from `L2DQNPolicyConfig` defaults or are otherwise notable.
    """
    defaults = L2DQNPolicyConfig()
    pk = policy.policy_kwargs
    tag = [
        get_tag_value(pk, 'buffer_size', 'rb={}', '', transform_value=f2s),
        get_tag_value(pk, 'learning_rate', 'lr={}', ''),
        get_tag_value(pk, 'timesteps_per_collection', 'tpc={}', '', 
                      check_value=lambda x: x if x != defaults.timesteps_per_collection else False),
        get_tag_value(pk, 'update_after', 'upaf={}', '', check_value=lambda x: x, transform_value=f2s),
        get_tag_value(pk, 'reg_coef', 'rcoef={}', '', transform_value=f2s),
    ]
    
    # Reset tags
    tag.extend([
        get_tag_value(pk, 'reset_buffer', 'rsrb','', check_value=lambda x: x if x == True else False),
        get_tag_value(pk, 'reset_optimizer', 'rsopt', '', check_value=lambda x: x if x == True else False),
    ])

    return '_'.join(filter(None, tag)) 


class L2DQNPolicyConfig(DQNPolicyConfig):
    """Hyperparameter/config container for `L2DQNPolicy`, extending `DQNPolicyConfig`.

    Attributes:
        reg_coef: Scaling for the L2 weight regularization term (i.e. lambda).
    """

    def __init__(self):
        super().__init__()
        self.reg_coef: float = 100000.0