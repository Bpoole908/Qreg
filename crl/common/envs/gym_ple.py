"""
    Code based on https://github.com/lusob/gym-ple/blob/master/gym_ple/ple_env.py
"""
import os
import importlib
from pdb import set_trace

import pygame
import gym
import numpy as np
from gym import spaces
from ple import PLE

class FakeALEInterface():
    """Minimal stand-in for ALE's interface, exposing just `lives()` backed by a PLE instance.

    Some code expects an Atari Learning Environment (ALE) interface (e.g. an
    `ale` attribute with a `lives()` method); PLE doesn't provide one even
    though it otherwise mimics ALE, so this wraps a `PLE` instance to supply it.
    """

    def __init__(self, ple):
        """
        Args:
            ple: The `ple.PLE` instance to forward `lives()` calls to.
        """
        self.ple = ple

    def lives(self):
        """Returns the number of lives remaining, per the wrapped PLE instance."""
        return self.ple.lives()

class GymPLE(gym.Env):
    """Gym environment wrapper around a PyGame Learning Environment (PLE) game.

    Adapted from https://github.com/lusob/gym-ple/blob/master/gym_ple/ple_env.py.
    Handles headless (SDL dummy driver) rendering, reviving a PLE game's PyGame
    surface after it was paused (needed for continual-learning setups that
    interleave multiple tasks/games in one process), and translating between
    PLE's action/observation conventions and Gym's `Env` API.
    """

    metadata = {'render_modes': ['rgb_array']}

    def __init__(
        self,
        game_id='FlappyBird',
        game_kwargs=None,
        ple_kwargs=None
    ):
        """
        Args:
            game_id: Name of a PLE game class (looked up under
                `ple.games.<game_id.lower()>`), or a game class/callable itself.
            game_kwargs: Optional kwargs passed to the game's constructor.
            ple_kwargs: Optional kwargs passed to `ple.PLE`'s constructor.
        """
        self.game_id = game_id
        self.game_kwargs = {} if game_kwargs is None else game_kwargs
        self.ple_kwargs = {} if ple_kwargs is None else ple_kwargs
        self.init()

    @property
    def _n_actions(self):
        """int: Number of discrete actions available in `self._action_set`."""
        return len(self._action_set)

    @property
    def ale(self):
        """FakeALEInterface: For Gym compatibility since PLE lacks an `ale` attribute even though PLE mimics the ALE interface.

        Ref: https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/src/ale_interface.cpp
        """
        return self.ple

    def init(self):
        """Constructs the underlying PLE game/instance and derives the Gym spaces.

        Sets `self.game`, `self.ple`, `self._action_meanings`/`_action_set`, and
        `self.action_space`/`self.observation_space`. Configures headless
        rendering via the SDL dummy video driver unless `display_screen` was
        requested in `ple_kwargs`.
        """
        if not self.ple_kwargs.get('display_screen',  False):
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        if isinstance(self.game_id, str):
            game_module_path = f'ple.games.{self.game_id}'.lower()
            game_module = importlib.import_module(game_module_path)
            self.game = getattr(game_module, self.game_id)(**self.game_kwargs)
        else:
            self.game = self.game_id(**self.game_kwargs)
        
        if pygame.display.get_init():
            pygame.display.quit()
        self.ple = PLE(self.game, **self.ple_kwargs)
        
        self._action_meanings, self._action_set  = self._build_action_info()
        self.screen_height, self.screen_width = self.ple.getScreenDims()
    
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3), dtype = np.uint8)

    # TODO: Gym 0.26.0 > returns info on reset
    def reset(self, seed=None, options=None):
        """Resets the PLE game and returns the initial observation.

        Args:
            seed: Optional RNG seed to apply to the PLE game before resetting.
            options: Unused; accepted for Gym `Env.reset` API compatibility.

        Returns:
            The initial observation (RGB frame) as a NumPy array.
        """
        self._check_headless_mode()
        self._revive_dead_screen()

        super().reset(seed=seed)
        if seed is not None:
            self._seed(seed)
        
        self.ple.reset_game()
        obs = self._get_obs()
        
        return obs
            
    def step(self, action):
        """Applies a discrete action to the PLE game and returns the transition.

        Args:
            action: Index into `self._action_set` (i.e. Gym's `action_space`).

        Returns:
            Tuple `(obs, reward, done, info)` where `obs` is the next RGB frame,
            `reward` is the PLE reward, `done` is whether the game ended, and
            `info` is always an empty dict.
        """
        self._check_headless_mode()
        self._revive_dead_screen()

        reward = self.ple.act(self._action_set[action])
        obs = self._get_obs()
        done = self.ple.game_over()
        return obs, reward, done, {}

    def _check_headless_mode(self):
        """Ensures the SDL video driver setting matches this game's `display_screen` setting.

        Note:
            If running back-to-back PyGame games with a mix of headless and
            video display in the same process, this makes sure the correct
            driver setup is active for the current task.
        """
        if not self.ple.display_screen and os.environ.get('SDL_VIDEODRIVER') != 'dummy':
            if pygame.display.get_init():
                pygame.display.quit()
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        elif self.ple.display_screen and 'SDL_VIDEODRIVER' in os.environ:
            if pygame.display.get_init():
                pygame.display.quit()
            os.environ.pop("SDL_VIDEODRIVER")
            
    def _revive_dead_screen(self):
        """Re-initializes this game's PyGame surface if it has been torn down.

        Note:
            A limitation of PyGame (i.e., PLE) is that only 1 Surface can act as
            the main screen at a given time for a given process. Thus, if a task
            is paused and others are run (i.e., continual evaluation) then the
            paused game's screen will die. This revives a dead surface by
            re-initializing it.
        """
        if not pygame.display.get_init():
            self.game._setup()

    def render(self, mode='human'):
        """Returns the current RGB frame.

        Args:
            mode: Unused; accepted for Gym `Env.render` API compatibility.

        Returns:
            The current observation (RGB frame) as a NumPy array.
        """
        # TODO: Not completed, need to do entire pygame rendering here
        #       and set PLE rendering to always false.
        img = self._get_obs()
        return img

    def _get_obs(self):
        """Reads the current PyGame screen surface as an (H, W, 3) RGB array."""
        # PLE rotates images so width and height need to be switched
        return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.game.screen)), axes=(1, 0, 2)
            )

    def _seed(self, seed):
        """Seeds the PLE game's RNG.

        Args:
            seed: Integer seed for `np.random.RandomState`.
        """
        rng = np.random.RandomState(seed)
        self.ple.rng = rng
        self.ple.game.rng = self.ple.rng

    def _build_action_info(self):
        """Builds Gym-ordered action meanings and the corresponding PLE action set.

        PLE's action set has NOOP last (if present); Gym convention expects
        NOOP first, so it is reordered here.

        Returns:
            Tuple `(action_meanings, action_set)`: parallel lists of action name
            strings and the corresponding PLE key codes (with `None` for NOOP).
        """
        action_meanings = ["NOOP"] if self.ple.add_noop_action else []
        action_set = [None] if self.ple.add_noop_action else []
        for action, key_id in self.game.actions.items():
            if action == 'NOOP':
                continue
            action_meanings.append(action.upper())
            action_set.append(key_id)
            
        return action_meanings, action_set
            
    def get_action_meanings(self):
        """Returns the list of human-readable action names, in Gym action-index order."""
        return self._action_meanings

    def close(self):
        """Tears down the PyGame display and quits PyGame, clearing the game's screen reference."""
        if self.game.screen is not None:
            # Disable game screen in case object needs pickling
            self.game.screen = None
            pygame.display.quit()
            pygame.quit()
            