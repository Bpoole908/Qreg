from pdb import set_trace

from torch.nn import functional as F    

from crl.policies.dqn.model import DQN
from crl.common.buffers.replay_buffers import Samples


def kld(input, target, gamma=1, log_target=False):
    """Computes a scaled batch-mean KL-divergence loss.

    Args:
        input: Log-probabilities (or probabilities if `log_target` refers only
            to `target`) to compare against `target`.
        target: Target distribution, expected in log-space if `log_target` is True.
        gamma: Scale factor applied to the resulting loss.
        log_target: Whether `target` is already in log-space (passed to `F.kl_div`).

    Returns:
        The scaled KL-divergence loss (batch-mean reduction).
    """
    return gamma * F.kl_div(
        input=input,
        target=target,
        reduction='batchmean',
        log_target=log_target
    )


def mse(input, target, gamma=1):
    """Computes a scaled mean-squared-error loss.

    Args:
        input: Predicted values.
        target: Target values.
        gamma: Scale factor applied to the resulting loss.

    Returns:
        The scaled MSE loss (mean reduction).
    """
    return gamma * F.mse_loss(input=input, target=target, reduction='mean')


def regularization(pred, target, gamma: float, use_kld: bool = False):
    """Computes a regularization loss between `pred` and a detached `target`, via KL-divergence or MSE.

    Args:
        pred: Predicted values (current model output).
        target: Target values to regularize towards (e.g. cached rehearsal
            outputs); detached before use so gradients don't flow into it.
        gamma: Scale factor applied to the resulting loss.
        use_kld: If True, applies `kld` to the log-softmax of `pred`/`target`
            (for logits/distributions); if False, applies `mse` directly.

    Returns:
        The scaled regularization loss.
    """
    target = target.detach()
    if use_kld:
        loss = kld(
            input=F.log_softmax(pred, dim=1), 
            target=F.log_softmax(target, dim=1), 
            gamma=gamma,
            log_target=True,
        )
    else:
        loss = mse(input=pred, target=target, gamma=gamma)

    return loss

    
class DRDQNLoss():
    """Computes the "qreg" data-rehearsal regularization loss: Q-value, Q-head-only, and embedding penalties against cached rehearsal outputs.

    This is the core loss of the "qreg" continual-learning method: instead of
    (or in addition to) storing raw past transitions, cached model outputs
    (embeddings, Q-values) for rehearsal samples are stored and the current
    model is regularized to reproduce them, mitigating forgetting without a
    full replay of past transitions.
    """

    def __init__(
        self,
        lambda_q: float = 1.0,
        lambda_e: float = 0,
        lambda_qhead: float = 0,
        soft_embeddings: bool = False,
        soft_q_values: bool = False
    ):
        """
        Args:
            lambda_q: Scaling factor for full-model Q-value regularization
                (Eq. 1 in the qreg paper). Set to 0 to disable.
            lambda_e: Scaling factor for embedding regularization. Set to 0 to disable.
            lambda_qhead: Scaling factor for Q-head-only regularization
                (gradient stops at the embedding). Set to 0 to disable.
            soft_embeddings: If True, computes the embedding regularization
                loss using KL-divergence; if False, uses MSE.
            soft_q_values: If True, computes the Q-value regularization losses
                using KL-divergence; if False, uses MSE.
        """
        self.lambda_qhead = lambda_qhead
        self.lambda_e = lambda_e
        self.lambda_q = lambda_q
        self.soft_embeddings = soft_embeddings
        self.soft_q_values = soft_q_values
        
    def __call__(
        self,
        dqn: DQN,
        rehearsal_samples: Samples,
    ):
        """Computes the combined embedding/Q-head/Q-value regularization loss for a batch of rehearsal samples.

        Args:
            dqn: The current `DQN` model, evaluated on the rehearsal samples'
                observations to get current embeddings/Q-values.
            rehearsal_samples: A `Samples` (with extra `embed`/`q_values`
                fields) batch drawn from the rehearsal replay buffer, holding
                cached embeddings/Q-values to regularize towards.

        Returns:
            Tuple `(loss, logs)`: the combined scalar loss, and a list of
            Tensorboard-style log dicts for the Q-diff, embedding, and
            (full-model) Q-value regularization loss components.

        Raises:
            AssertionError: If a cached rehearsal field's shape doesn't match
                the corresponding current model output's shape, for any
                enabled regularization term.
        """
        embed_loss = 0
        q_diff_loss = 0
        bc_loss = 0
        
        embed, q_values = dqn(rehearsal_samples.observations)

        # Embedding Loss
        if self.lambda_e != 0:
            assert rehearsal_samples.embed.shape == embed.shape
            embed_loss = regularization(
                pred=embed, 
                target=rehearsal_samples.embed,
                gamma=self.lambda_e,
                use_kld=self.soft_embeddings,
            )
            # assert embed_loss == self.lambda_e*(embed - rehearsal_samples.embed.detach()).pow(2).mean()
            
        # ONLY Q-Head Reg Loss 
        if self.lambda_qhead != 0:
            # Get detached Q-values so propagation stops at head
            detached_q_values = dqn.q_head(embed.detach())
            assert rehearsal_samples.q_values.shape == detached_q_values.shape
            q_diff_loss = regularization(
                pred=detached_q_values, 
                target=rehearsal_samples.q_values, 
                gamma=self.lambda_qhead, 
                use_kld=self.soft_q_values
            )
            # assert q_diff_loss == self.lambda_qhead*(detached_q_values - rehearsal_samples.q_values.detach()).pow(2).mean()
        
        # Qreg loss (Eq. 1 in paper)
        if self.lambda_q != 0:
            assert rehearsal_samples.q_values.shape == q_values.shape
            bc_loss = regularization(
                pred=q_values, 
                target=rehearsal_samples.q_values, 
                gamma=self.lambda_q, 
                use_kld=self.soft_q_values
            )
            # assert bc_loss == self.lambda_q*(q_values - rehearsal_samples.q_values.detach()).pow(2).mean()

        loss =  embed_loss + q_diff_loss + bc_loss
        
        logs = [
            {"type": "scalar", "tag": "q_diff_loss", "value": q_diff_loss},
            {"type": "scalar", "tag": "embed_loss", "value": embed_loss},
            {"type": "scalar", "tag": "bc_loss", "value": bc_loss},
        ]

        return loss, logs