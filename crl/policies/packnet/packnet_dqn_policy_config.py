from crl.policies.dqn.dqn_policy_config import DQNPolicyConfig, get_tag_value
from crl.common.yaml import convert_float_to_str as f2s


def experiment_tag(policy, exp):
    """Builds a short experiment tag string summarizing a PackNet-DQN policy's non-default hyperparameters.

    Args:
        policy: The policy instance, whose `policy_kwargs` dict is inspected.
       exp: The raw experiment config.

    Returns:
        An underscore-joined tag string, including only the hyperparameters
        that differ from `PackNetDQNPolicyConfig` defaults or are otherwise notable.
    """
    defaults = PackNetDQNPolicyConfig()
    pk = policy.policy_kwargs
    tag = [
        get_tag_value(pk, 'buffer_size', 'rb={}', '', transform_value=f2s),
        get_tag_value(pk, 'learning_rate', 'lr={}', ''),
        get_tag_value(pk, 'timesteps_per_collection', 'tpc={}', '', 
                      check_value=lambda x: x if x != defaults.timesteps_per_collection else False),
        get_tag_value(pk, 'update_after', 'upaf={}', '', check_value=lambda x: x, transform_value=f2s),
        get_tag_value(pk, 'retrain_steps', 'steps={}', '', transform_value=f2s),
    ]
    
    # Reset tags
    tag.extend([
        get_tag_value(pk, 'reset_buffer', 'rsrb','', check_value=lambda x: x if x == True else False),
        get_tag_value(pk, 'reset_optimizer', 'rsopt', '', check_value=lambda x: x if x == True else False),
    ])
        
    return '_'.join(filter(None, tag)) 


class PackNetDQNPolicyConfig(DQNPolicyConfig):
    """Hyperparameter/config container for `PackNetDQNPolicy`, extending `DQNPolicyConfig`.

    Attributes:
        retrain_steps: How many steps to retrain once weights have been pruned.
        num_tasks: Total number of tasks in one cycle; must be set to an int
            before use (determines the per-task pruning percentage and when
            `on_first_cycle` becomes False).
    """

    def __init__(self):
        super().__init__()
        self.retrain_steps: int = 10000
        self.num_tasks = None
