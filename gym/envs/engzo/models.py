import uuid
import numpy as np

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
        return reduce(list.__add__, [grp.knowledges for grp in self.group.preliminaries])

class Activity(BaseModel):
    """
    Activity may contain questions and answers,
    it attached one or more knowledges
    """
    def __init__(self, kg):
        self.knowledges = [kg]

    def append_knowledge(self, kg):
        if kg not in self.knowledges:
            return self.knowledges.append(kg)
