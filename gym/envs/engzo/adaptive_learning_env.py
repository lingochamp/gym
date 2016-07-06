import numpy as np
import gym
import logging

from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import colorize, seeding
# from gym.envs.toy_text import discrete
from gym.envs.engzo import models

from six import StringIO

logger = logging.getLogger(__name__)

class AdaptiveLearningEnv(gym.Env):

    metadata = { 'render.modes': ['human', 'ansi'] }
    ratio = 5

    def __init__(self):
        self.action_space = models.generate_activites(self.ratio)
        # self.observation_space = Box()
        self.reward_range = (0, 1)
        self._seed()
        self._reset()


    def _step(self, action):
        assert self.action_space.contains(action)
        raise NotImplemented
        # return self._get_obs(), reward, done, {}

    def _get_obs(self):
        raise NotImplemented

    def _reset(self):
        return self._get_obs()


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render(self, mode='human', close=False):
        if close:
            return
        # outfile = StringIO() if mode == 'ansi' else sys.stdout
        pass
