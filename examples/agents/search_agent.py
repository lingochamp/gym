import logging
import os, sys
import numpy as np

import gym
from gym.envs.engzo.models import Knowledge, Activity, KnowledgeGroup


class SearchAgent(object):
    def __init__(self, actions, knowledges, simulator):
        self.__actions = actions
        self.__knowledges = knowledges
        self.__simulator = simulator

    def act(self, observation):
        best_action, best_reward = None, 0.

        for i, a in enumerate(self.__actions):
            action = Activity(a, self.__knowledges)
            ob, reward, done = self.__simulator.progress(observation, action)
            if reward > best_reward or best_action is None:
                best_action = i
                best_reward = reward

        assert best_action is not None

        return best_action


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    env = gym.make('EngzoAdaptiveLearning-v0')
    env.monitor.start('/tmp/engzo-agent-results', force=True)

    agent = SearchAgent(env.activities, env.knowledges, env.simulator)

    max_episode, max_step = 100, 1000

    for episode in range(max_episode):
        ob = env.reset()
        reward_tot = 0.

        for step in range(max_step):
            action = agent.act(ob)
            ob, reward, done, _ = env.step(action)
            reward_tot += reward
            print step, sum(ob)

            if done:
                break

        print "new state", sum(ob), reward_tot / max_step

    env.monitor.close()

