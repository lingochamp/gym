import logging
import os, sys
import numpy as np

import gym
from gym.envs.engzo.models import Knowledge, Activity, KnowledgeGroup
from engzo_agent import BaseAgent

class SequenceAgent(BaseAgent):
    def __init__(self, sorted_actions, max_steps):
        self._sorted_actions = sorted_actions
        self._next_action = 0
        BaseAgent.__init__(self, max_steps)

    def act(self, observation, reward, done):
        current_action = self._sorted_actions[self._next_action]
        self._next_action += 1
        if self._next_action == len(self._sorted_actions):
            self._next_action = 0
        return current_action

class SequenceAgent2(object):
    def __init__(self, sorted_actions):
        self._sorted_actions = sorted_actions
        self._next_action = 0

    def act(self, observation, reward, done):
        current_action = self._sorted_actions[self._next_action]
        self._next_action += 1
        if self._next_action == len(self._sorted_actions):
            self._next_action = 0
        return current_action

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
    knowledges = env.knowledges
    activities = env.activities
    num_action = len(activities)
    activities_tuple = [(idx,
                         list(activities[idx]).index(max(activities[idx])),
                         knowledges[list(activities[idx]).index(max(activities[idx]))].level(),
                         knowledges[list(activities[idx]).index(max(activities[idx]))].group)
                        for idx in range(num_action)]

    sorted_tuple = [a[0] for a in sorted(activities_tuple, key=lambda tup: (tup[2], tup[1], tup[3]))]
    episode_count = 100
    max_steps = 1000
    reward = 0
    done = False
    agent = SequenceAgent(sorted_tuple, max_steps)

    for i in range(episode_count):
        ob = env.reset()
        total_reward = 0
        agent.viewer.geoms = []

        for j in range(max_steps):
            # env.render()
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            total_reward += reward
            agent.draw_state(j, sum(ob))
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        print("new state", sum(ob), total_reward / max_steps)
    # Dump result info to disk
    env.monitor.close()
