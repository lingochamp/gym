from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import colorize, seeding
from gym.envs.toy_text import discrete

from six import StringIO

import numpy as np
import logging

class AdaptiveLearningEnv(discrete.DiscreteEnv):
    metadata = { 'render.modes': ['human', 'ansi'] }

    def __init__(self):
        self.action_space = Discrete(200)
        # self.observation_space = Box()
        self.reward_range = (0, 1)
        pass

    def _step(self, action):
        pass

    def _reset(self):
        pass

    def _render(self, mode='human', close=False):
        if close:
            return
        # outfile = StringIO() if mode == 'ansi' else sys.stdout
        pass
