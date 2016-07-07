import gym
import logging
import os
import pickle
import numpy as np

from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import colorize, seeding

from operator import attrgetter

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

    def _configure(self):
        self._load_activities()
        self.action_space = Discrete(len(self.activities))
        self.observation_space = Box(0, 1, len(self.knowledges)) #
        self.simulator = StudentSimulator()

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
        self.ob = Box(0.1, 0.1, len(self.knowledges)).sample()
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
        radius = 10
        init_alpha = 0.1
        margin = radius * 2.5
        max_per_line = (screen_width - margin * 2) / margin

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            colors = np.array([[78,191,126], [254,178,45], [175,101,194]])/255.

            for (i, x) in enumerate(sorted(self.knowledges, key=attrgetter('level', 'group'), reverse=True)):
                h, w = divmod(i, max_per_line)
                w = screen_width - 20  - w * (margin + 10)
                h = screen_height - 20 - h * (margin + 10)
                t = self.viewer.draw_circle(radius)
                t.add_attr(rendering.Transform((w, h)))
                r, g, b = colors[x.level() - 1]
                t.set_color(r, g, b, init_alpha)
                self.viewer.add_geom(t)

        #TODO: Update state here
        return self.viewer.render()
