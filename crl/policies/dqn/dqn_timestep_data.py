from continual_rl.policies.timestep_data_base import TimestepDataBase

class DQNTimestepData(TimestepDataBase):
    """Carries the observation/action of one timestep forward so `DQNPolicy` can build the next replay-buffer transition."""

    def __init__(self, observation, action):
        """
        Args:
            observation: The observation at this timestep.
            action: The action taken at this timestep.
        """
        super().__init__()
        self.observation = observation
        self.action = action