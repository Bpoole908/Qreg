from pdb import set_trace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from crl.policies.dqn.model import DQN

 
class PackNetDQN(DQN):
    """DQN variant with a COOM-style embedding layer (extra linear/norm/activation stack), used by `PackNetDQNPolicy`.

    Cite:
        Tomilin, T., Fang, M., Zhang, Y. & Pechenizkiy, M. COOM: A Game
        Benchmark for Continual Reinforcement Learning. Advances in Neural
        Information Processing Systems 36, 67794-67832 (2023).
    """

    def build_embedding_layer(self, n_flattened):
        """Builds the COOM-style embedding stack: linear -> layer norm -> tanh -> linear -> leaky ReLU.

        Overrides `DQN.build_embedding_layer`'s single linear + activation
        with an extra intermediate 256-unit layer normalized/tanh-activated
        block, matching COOM's architecture.

        Args:
            n_flattened: Number of flattened conv output features (input size
                of the first linear layer).

        Returns:
            An `nn.Sequential` implementing the embedding stack.
        """
        return nn.Sequential(
            nn.Linear(n_flattened, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, self.embedding_size),
            nn.LeakyReLU(negative_slope=0.2),
        )