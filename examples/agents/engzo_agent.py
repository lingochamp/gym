import logging
import os, sys
import numpy as np

import gym
from gym.envs.engzo.models import Knowledge, Activity, KnowledgeGroup
from gym.envs.classic_control import rendering

class BaseAgent(object):
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.half_width = 300 if self.max_steps > 300 else self.max_steps
        self.step = int(self.max_steps / 2 / self.half_width)
        self.height = 200
        self.viewer = rendering.Viewer(self.half_width * 2, self.height)

    def act(self, observation, reward, done):
        pass

    def draw_state(self, j, state):
        if j % self.step == 0:
            point = self.viewer.draw_circle(2)
            line = self.viewer.draw_polyline([(j / self.step, 0), (j / self.step, self.height)])
            line.set_linewidth(0.1)
            point.add_attr(rendering.Transform((j / self.step, 8 + state)))
            point.set_color(0, 0, 0, 0.8)
            line.set_color(0, 0, 0, 0.5)
            self.viewer.add_geom(point)
            self.viewer.add_geom(line)
            self.viewer.render(return_rgb_array=True)

class EngzoAgent(BaseAgent):
    def __init__(self, action_space, max_steps):
        BaseAgent.__init__(self, max_steps)
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    env = gym.make('EngzoAdaptiveLearning-v0')
    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    env.monitor.start('/tmp/engzo-agent-results', force=True)

    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.

    episode_count = 100
    max_steps = 1000
    reward = 0
    done = False

    agent = EngzoAgent(env.action_space, max_steps)

    for i in range(episode_count):
        ob = env.reset()
        agent.viewer.geoms = []

        for j in range(max_steps):
            env.render()
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            agent.draw_state(j, sum(ob))
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
            print(sum(ob), reward*1000)
    # Dump result info to disk
    env.monitor.close()
