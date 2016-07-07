import numpy as np
import gym
import logging

from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import colorize, seeding
# from gym.envs.toy_text import discrete

from gym.envs.engzo import models
from gym.envs.engzo.student_simulator import StudentSimulator

from six import StringIO

logger = logging.getLogger(__name__)

class AdaptiveLearningEnv(gym.Env):

    metadata = { 'render.modes': ['human', 'ansi'] }
    ratio = 5

    def __init__(self):
        self.knowledges = models.generate_knowledges()
        self.action_space = models.ActivitySpaceWrapper(models.generate_activities(self.knowledges, self.ratio))
        self.observation_space = Box(0, 1, len(self.knowledges))
        self.reward_range = (0, 1)
        self.simulator = StudentSimulator(self.action_space)
        self.viewer = None
        self.ob = None
        self._seed()
        self._reset()

        # Just need to initialize the relevant attributes
        self._configure()


    def set_data(self, ks=None, action_space=None):
        if ks is not None: self.knowledges = ks
        if action_space is not None: self.action_space = action_space

    def _configure(self, display=None):
        self.display = display

    def _step(self, action):
        assert self.action_space.contains(action)
        a = models.Activity(action)
        ob, reward, done = self.simulator.progress(a.knowledge, a)
        self.ob = ob
        return ob, reward, done, {}

    def _reset(self):
        self.ob = Box(0.05, 0.1, len(self.knowledges)).sample()
        return self.ob

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
