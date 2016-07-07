import gym
import logging
import os
import pickle
import numpy as np

from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import colorize, seeding
# from gym.envs.toy_text import discrete

from gym.envs.engzo.models import Knowledge, Activity, KnowledgeGroup
from gym.envs.engzo.student_simulator import StudentSimulator

from six import StringIO

logger = logging.getLogger(__name__)

class AdaptiveLearningEnv(gym.Env):

    metadata = { 'render.modes': ['human', 'ansi'] }

    def __init__(self, filename='activities.pkl'):
        self.filename = filename
        self.assets_dir = os.path.dirname(os.path.abspath(__file__))
        self.reward_range = (0, 1)
        self.viewer = None
        self.ob = None
        self._configure()
        self._seed()
        self._reset()

    def _configure(self, display=None):
        self.display = display
        self._load_activities()
        self.action_space = Discrete(len(self.activities))
        self.observation_space = Box(0, 1, len(self.knowledges)) #
        self.simulator = StudentSimulator(self.action_space)

    def _load_activities(self):
        data_file = os.path.join(self.assets_dir, 'assets/%s' % self.filename)
        pkl_file = open(data_file, 'rb')
        self.knowledges = pickle.load(pkl_file)
        self.activities = pickle.load(pkl_file)
        pkl_file.close()

    def _step(self, action):
        assert self.action_space.contains(action)
        a = Activity(self.activities[action], self.knowledges)
        ob, reward, done = self.simulator.progress(self.ob, a)
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
