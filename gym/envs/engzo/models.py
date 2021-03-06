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

    def __init__(self, _id=None):
        self._id = _id


class KnowledgeGroup(BaseModel):
    def __init__(self, level, kg_num, preliminaries=None):
        self.knowledges = [Knowledge(self) for i in range(kg_num)]
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
        return [x for x in self.group.knowledges if x is not self]

    def preliminaries(self):
        if self.group.preliminaries:
            return reduce(list.__add__, [grp.knowledges for grp in self.group.preliminaries])
        else:
            return []


class Activity(BaseModel):
    """
    Activity is a wrapper class for action vector
    """

    def __init__(self, psi, ks):
        """
        Args:
            psi (:obj:`ndarray`): Requirement of each knowledge.
            ks (list of :obj:`Knowledge`): All knowledge.
        """
        self.psi = psi
        self.knowledge_indexes = np.nonzero(psi)
        self.knowledges = [ks[i] for i in self.knowledge_indexes[0]]
        self.related_knowledge_indexes = self.__related_knowledge_indexes()
        self.preliminary_knowledge_indexes = self.__preliminary_knowledge_indexes()

    def __related_knowledge_indexes(self):
        """
        Returns: Index of self + knowledges in same group
        """
        ks = set(self.knowledges)
        for k in self.knowledges:
            ks.update(k.sibling())
        return np.array([k._id for k in ks])

    def __preliminary_knowledge_indexes(self):
        ks = set()
        for k in self.knowledges:
            ks.update(k.preliminaries())
        return [k._id for k in ks]


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
    d = np.random.multinomial(kg_max, [5 / 10., 3.5 / 10., 1.5 / 10.])
    groups = reduce(list.__add__, [_generate_groups(i + 1, x, group_kg_max) for i, x in enumerate(d)])
    for group in groups:
        if group.level > 1:
            gs = [g for g in groups if g.level == group.level - 1]
            group.preliminaries = random.sample(gs, Discrete(len(gs)).sample())
    ks = reduce(list.__add__, [x.knowledges for x in groups])
    for i, x in enumerate(ks): x._id = i
    return ks


def _generate_activities(ks, num_activity_per_knowledge):
    activities = []
    K = len(ks)

    # Each activity attaches exactly 1 main knowledge and 0 to 2 minor knowledges
    for main_knowledge in ks:
        psi_bin_size = 1. / num_activity_per_knowledge

        for i in range(num_activity_per_knowledge):
            act_psi = np.zeros(K)

            # Set main knowledge
            psi_low, psi_high = max(.1, i * psi_bin_size), min(1., (i + 1) * psi_bin_size)
            act_psi[main_knowledge._id] += np.random.uniform(psi_low, psi_high)

            # Set minor knowledge
            # Choose minor knowledge from knowledge in same level or preliminary level group
            # Random from 0. to main knowledge psi
            knowledge_candidates = [k for k in ks if
                                    k.group.level <= main_knowledge.group.level and k != main_knowledge]
            num_minor_knowledge = random.randint(0, 2)
            assert len(knowledge_candidates) >= num_minor_knowledge
            for k in random.sample(knowledge_candidates, num_minor_knowledge):
                act_psi[k._id] += np.random.uniform(0., act_psi[main_knowledge._id])

            activities.append(act_psi)

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
