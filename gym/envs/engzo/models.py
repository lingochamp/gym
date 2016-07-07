import random
import pickle
import os
import numpy as np

from gym import Space
from gym.spaces import Discrete

class BaseModel(object):
    """
    BaseModel for engzo Adaptive Learning Env
    """
    def __init__(self):
        pass

class KnowledgeGroup(BaseModel):
    def __init__(self, level, kg_num, preliminaries=None):
        self.knowledges = [ Knowledge(self) for i in range(kg_num) ]
        self.preliminaries = preliminaries
        self.level = level

class Knowledge(BaseModel):
    """
    Knowledge is an unique and specific entry to learn
    """
    def __init__(self, group=None):
        self.group = group

    def add_to(self, group):
        if group not in self.groups:
            self.groups.append(group)

    def level(self):
        return self.group.level

    def sibling(self):
        result = []
        for x in self.group.knowledges:
            if x is not self:
                result.append(x)
        return result

    def preliminaries(self):
        if self.group.preliminaries is not None:
            return reduce(list.__add__, [grp.knowledges for grp in self.group.preliminaries])
        else:
            return []

class Activity(BaseModel):
    """
    Activity is a wrapper class for action vector
    """

    def __init__(self, activity):
        self.activity = activity

    def knowledges(self, ks):
        """
        Return all knowledges of activity,
        The first is main knowledge
        """
        m = max(self.activity)
        ary = []
        indexes = sorted([(j, i) for i, j in enumerate(self.activity) if j != 0])
        for (x, i) in indexes:
            ary.append(ks[i])
        return ary

class ActivitySpaceWrapper(Space):
    """
    A ActivitySpaceWrapper for ITS,
    It's a 2-D array of knowledges
    """
    def __init__(self, spaces):
        self.spaces = spaces

    def sample(self):
        return self.spaces[np.random.randint(len(self.spaces))]

    def contains(self, x):
        return x in self.spaces


##-------------------- Help Methods --------------------
def _generate_groups(level, kg_max=200, group_kg_max=5):
    """
    Generate Knowledge Groups

    Args:
       level (Int): The difficulty of knowledge group
       kg_max (Int): The maximum of total knowledges
       group_kg_max (Int): The maximum knowledge number in a knowledge group
    """
    remain = kg_max
    groups = []
    for i in range(kg_max):
        if remain > 0:
            n = Discrete(group_kg_max).sample() + 1
            groups.append(KnowledgeGroup(level, n))
            remain -= n
    return groups

def _generate_knowledges(kg_max=200, group_kg_max=5.):
    d = np.random.multinomial(kg_max, [5/10., 3.5/10., 1.5/10.])
    groups = reduce(list.__add__, [_generate_groups(i+1, x, group_kg_max) for i, x in enumerate(d)])
    for group in groups:
        if group.level > 1:
            gs = [g for g in groups if g.level == group.level - 1]
            group.preliminaries = random.sample(gs, Discrete(len(gs)).sample())
    return reduce(list.__add__, [x.knowledges for x in groups])

def _generate_activities(ks, ratio):
    ## generate activities
    activities = []
    for index, k in enumerate(ks):
        for i in range(ratio):
            act = np.zeros(len(ks))
            act[index] += np.random.uniform(0.5, 1.)
            if act[index] < 0: act[index] = 0.5
            siblings = k.sibling()
            np.random.shuffle(siblings)
            for s in siblings[0:2]:
                ki = ks.index(s)
                act[ki] += np.random.uniform(0, 0.5)
                if act[ki] < 0: act[ki] = 0
            activities.append(act)
    return np.asarray(activities)

def main():
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/activities.pkl')
    output = open(data_file, 'wb')
    ks = _generate_knowledges()
    acts = _generate_activities(ks, 5)
    pickle.dump(ks, output)
    pickle.dump(acts, output)
    output.close()

if __name__ == '__main__':
	main()