
from ple import games

class ContinualCatcher(games.Catcher):
  """A PLE Catcher variant whose fruit-fall speed is parameterized, for use as a continual-learning task.

  Varying `fall_speed_modifier` across tasks lets a single game act as a family
  of distinct-but-related tasks for continual/multi-task RL experiments.
  """

  def __init__(self, fall_speed_modifier, width=640, height=640, init_lives=3):
    """
    Args:
        fall_speed_modifier: The higher the number, the faster the fruits fall.
        width: Screen width in pixels.
        height: Screen height in pixels.
        init_lives: Number of lives the agent starts with.
    """
    super(ContinualCatcher, self).__init__(width, height, init_lives)
    # Taken from Catcher source code
    self.fruit_fall_speed =  (0.00095 * height) + 0.03*fall_speed_modifier

class ContinualFlappyBird(games.FlappyBird):
  """A PLE FlappyBird variant whose pipe gap is parameterized, for use as a continual-learning task.

  Varying `pipe_gap_modifier` across tasks lets a single game act as a family
  of distinct-but-related tasks for continual/multi-task RL experiments.
  """

  def __init__(self, pipe_gap_modifier, width=288, height=512, pipe_gap=100):
    """
    Args:
        pipe_gap_modifier: The higher the number, the smaller the gap between pipes.
        width: Screen width in pixels.
        height: Screen height in pixels.
        pipe_gap: Base gap size (in pixels) before applying `pipe_gap_modifier`.
    """
    pipe_gap = pipe_gap - 5*pipe_gap_modifier
    super(ContinualFlappyBird, self).__init__(width, height, pipe_gap)
