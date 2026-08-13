from typing import Union, Dict
from pdb import set_trace

from crl.policies.dqn.dqn_policy_config import DQNPolicyConfig, get_tag_value
from crl.common.yaml import convert_float_to_str as f2s


def experiment_tag(policy, task_seq):
    """Builds a short experiment tag string summarizing a DRDQN policy's non-default hyperparameters.

    Args:
        policy: The policy instance, whose `policy_kwargs` dict is inspected.
       task_seq: The raw task sequence config.

    Returns:
        An underscore-joined tag string, including only the hyperparameters
        that differ from `DRDQNPolicyConfig` defaults or are otherwise notable.
    """
    defaults = DRDQNPolicyConfig()
    pk = policy.policy_kwargs
    tag = [
        get_tag_value(pk, 'rrb_batch_size', 'rbs={}', '', check_value=lambda x: x if x != 0 else False),
        get_tag_value(pk, 'buffer_size', 'rb={}', '', transform_value=f2s),
        get_tag_value(pk, 'rrb_buffer_size', 'rrb={}', '', transform_value=f2s),
        get_tag_value(pk, 'learning_rate', 'lr={}', ''),
        get_tag_value(pk, 'rrb_add_frequency', 'raf={}', '', transform_value=f2s),
        get_tag_value(pk, 'rrb_add_history', 'rah={}', '', transform_value=f2s),
        get_tag_value(pk, 'rrb_add_sample_size', 'rass={}', '', check_value=lambda x: x, transform_value=f2s),
        get_tag_value(pk, 'rrb_update_frequency', 'ruf={}', '', transform_value=f2s),
        get_tag_value(pk, 'timesteps_per_collection', 'tpc={}', '', check_value=lambda x: x if x != defaults.timesteps_per_collection else False),
        get_tag_value(pk, 'wait_to_reg', 'wtr', '', check_value=lambda x: x),
        get_tag_value(pk, 'update_after', 'upaf={}', '', check_value=lambda x: x, transform_value=f2s),
    ]
    if 'rrb_loss_kwargs' in pk:
        rlk = policy.policy_kwargs.rrb_loss_kwargs
        tag.extend([
            get_tag_value(rlk, 'lambda_e', 'le={}', '', check_value=lambda x: x if x != 0 else False),
            get_tag_value(rlk, 'lambda_qhead', 'lqh={}', '', check_value=lambda x: x if x != 0 else False),
            get_tag_value(rlk, 'lambda_q', 'lq={}', '', check_value=lambda x: x if x != 0 else False),
    ])
        
    # Reset tags
    tag.extend([
        get_tag_value(pk, 'reset_buffer', 'rsrb','', check_value=lambda x: x if x == True else False),
        get_tag_value(pk, 'reset_optimizer', 'rsopt', '', check_value=lambda x: x if x == True else False),
    ])
    
    return '_'.join(filter(None, tag))   


class DRDQNPolicyConfig(DQNPolicyConfig):
    """Hyperparameter/config container for `DRDQNPolicy`, extending `DQNPolicyConfig`.

    Attributes:
        rrb_buffer_size: Size of the rehearsal replay buffer.
        rrb_batch_size: Number of rehearsal samples to draw per gradient
            step (capped to the buffer's current size if smaller), combined
            with the normal samples drawn from the replay buffer.
        wait_to_reg: If True, waits to apply regularization until after the
            first task ends (useful if `rrb_update_frequency` is less than the
            number of steps per task).
        rrb_update_frequency: Rate at which the rehearsal replay buffer has
            its samples updated. Updates are only applied to samples matching
            the current task id.
        rrb_add_frequency: Rate at which samples are added to the rehearsal
            replay buffer. When using memory optimization for the replay
            buffer, ensure this is always at least `rrb_add_history + 1` (e.g.
            `rrb_add_history=2000` requires an add frequency of 2001, to
            ensure 2000 samples and their next states are in the replay buffer).
        rrb_add_history: Number of latest samples to use from the replay
            buffer when adding new samples to the rehearsal replay buffer.
        rrb_add_sample_size: Number of samples selected (from `rrb_add_history`)
            each time samples are added to the rehearsal replay buffer.
        rrb_loss_kwargs: Kwargs for `DRDQNLoss`.
        iteratively_compute_samples: Whether to compute cached embeddings/
            Q-values for rehearsal samples in one pass or in smaller chunks
            (see `iterative_compute_size`).
        iterative_compute_size: Chunk size used when
            `iteratively_compute_samples` is True, for adding and updating
            rehearsal samples.
        log_sample_rate: Rate at which larger logs like histograms will be
            tracked. These can quickly inflate event-file space if done too frequently.
        networks_to_log: Names of sub-networks (within the DQN module) whose
            weights and gradients should be logged.
    """
    def __init__(self):
        super(DRDQNPolicyConfig, self).__init__()       
        # Policy
        self.rrb_buffer_size: int = 100000
        self.rrb_batch_size: int = 256
        self.wait_to_reg: bool = False
        self.rrb_update_frequency: Union[int, None] = None
        self.rrb_add_frequency: int = 2001
        self.rrb_add_history: Union[int, None] = None
        self.rrb_add_sample_size: int = 64
        self.rrb_loss_kwargs: Dict = {}
        self.iteratively_compute_samples: bool = False
        self.iterative_compute_size: int = 2000 # Used for adding and updating reg samples

        self.log_sample_rate = 0.005
        self.networks_to_log: list = None