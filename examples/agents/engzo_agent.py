import logging
import os, sys
import numpy as np

import gym
from gym.envs.engzo.models import Knowledge, Activity, KnowledgeGroup
from gym.envs.classic_control import rendering

class EngzoAgent(object):
    def __init__(self, action_space, max_steps):
        self.action_space = action_space
        m = 200 if max_steps > 200 else max_steps
        self.viewer = rendering.Viewer(m * 2, m)

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def draw_reward(self, j, reward):
        point = agent.viewer.draw_circle(2)
        line = agent.viewer.draw_polyline([(j * 2, 0), (j * 2, max_steps)])
        line.set_linewidth(0.1)
        point.add_attr(rendering.Transform((j * 2, 8 + reward * 1000)))
        point.set_color(0, 0, 0, 0.8)
        line.set_color(0, 0, 0, 0.5)
        agent.viewer.add_geom(point)
        agent.viewer.add_geom(line)
        agent.viewer.render(return_rgb_array=True)


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
            agent.draw_reward(j, reward)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
            print(sum(ob), reward*1000)
    # Dump result info to disk
    env.monitor.close()
