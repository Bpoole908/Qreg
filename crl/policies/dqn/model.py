from pdb import set_trace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

  
class DQN(nn.Module):
    """The Nature DQN CNN architecture: conv encoder -> embedding -> linear Q-value head.

    Note:
        Mnih, Volodymyr, et al. "Human-level control through deep reinforcement
        learning." Nature 518.7540 (2015): 529-533.
    """

    def __init__(
        self,
        observation_space,
        output_dims,
        flatten_dims=(1, 2),
        normalize_image=True,
        embedding_size=256,
        encoder_activation=nn.ReLU,
    ):
        """
        Args:
            observation_space: Gym space representing the input observation space.
            output_dims: Number of actions to output Q-values for.
            flatten_dims: Dimensions to flatten together during preprocessing
                (by default, stacked frames and color channels, assuming frame
                stacking is used and color channels aren't otherwise removed).
                Set to None or False to disable.
            normalize_image: If True, divides pixel values by 255 during
                preprocessing; otherwise just casts to float.
            embedding_size: Size of the embedding layer between the conv
                encoder and the Q-value head.
            encoder_activation: Activation module class used after each conv
                layer and the embedding layer. Defaults to `nn.ReLU` if None.
        """
        super(DQN, self).__init__()
        self.encoder_activation = nn.ReLU if encoder_activation is None else encoder_activation
        self.embedding_size = embedding_size
        self.output_dims = output_dims
        self.flatten_dims = flatten_dims
        self.normalize_image = normalize_image
        # Color's by stacked frames
        n_input_channels = observation_space.shape[0] * observation_space.shape[1]
        encoder = self.build_encoder(n_input_channels)
        # Compute output shape of the CNN
        with torch.no_grad():
            dummy_x = torch.as_tensor(observation_space.sample())
            # Add batch dimension and preprocess before passing to encoder to get output
            n_flattened = encoder(self.preprocessing(dummy_x[None])).shape[1]
            
        embedding = self.build_embedding_layer(n_flattened)
        self.encoder = nn.Sequential(
            *list(encoder),
            *list(embedding)
        )
        # Head for q-values
        self.q_head = self.build_q_value_head(embedding_size, output_dims)

    def build_encoder(self, n_input_channels):
      """Builds the Nature-DQN 3-layer conv stack (with activations) followed by a flatten.

      Args:
          n_input_channels: Number of input channels to the first conv layer.

      Returns:
          An `nn.Sequential` of conv/activation layers ending in `nn.Flatten()`.
      """
      return nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            self.encoder_activation(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            self.encoder_activation(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            self.encoder_activation(),
            nn.Flatten(),
        )
    
    def build_embedding_layer(self, n_flattened):
        """Builds the linear + activation layer projecting flattened conv features to the embedding space.

        Args:
            n_flattened: Number of flattened conv output features (input size
                of the linear layer).

        Returns:
            An `nn.Sequential` of one `nn.Linear` layer followed by an activation.
        """
        return nn.Sequential(
            nn.Linear(n_flattened, self.embedding_size),
            self.encoder_activation(),
        )

    def build_q_value_head(self, embedding_size, output_dims):
        """Builds the linear layer mapping embeddings to per-action Q-values.

        Args:
            embedding_size: Size of the input embedding.
            output_dims: Number of actions (output Q-values).

        Returns:
            An `nn.Sequential` containing a single `nn.Linear` layer.
        """
        return nn.Sequential(nn.Linear(embedding_size, output_dims))

    def preprocessing(self, x):
        """Flattens stacked-frame/color dims (if configured) and normalizes/casts pixel values.

        Args:
            x: Raw input observation batch.

        Returns:
            The preprocessed tensor, ready for the conv encoder.
        """
        if self.flatten_dims:
            x = torch.flatten(x, start_dim=self.flatten_dims[0], end_dim=self.flatten_dims[1])
            
        if self.normalize_image:
            x = x / 255.0
        else:
            x = x.to(torch.float)
            
        return x

    def forward(self, x):
        """Computes the embedding and Q-values for a batch of observations.

        Args:
            x: Raw input observation batch.

        Returns:
            Tuple `(embed, q_values)`: the embedding-layer output and the
            per-action Q-values.
        """
        x = self.preprocessing(x)
        embed = self.encoder(x)
        q_values = self.q_head(embed)

        return embed, q_values
