from models import Activity, Knowledge, KnowledgeGroup
import numpy as np
import logging

logger = logging.getLogger(__name__)


class StudentSimulator:
    def __init__(self, activities):
        """

        Args:
            activities (list of :obj:`Activity`): The whole activity bank.

        """
        pass

    def __performance(self, state, action):
        return 0.0

    def progress(self, state, action):
        """

        Args:
            state (:obj:`np.ndarray` of float): Knowledge mastery of student.
            action (:obj:`Activity`): Activity chosen by student.

        Returns:
            new_state (:obj:`np.ndarray` of float): Updated knowledge mastery.
            reward (float): Amount of reward achieved by the previous action.
            complete (bool): If student meet the criteria of completion.

        """
        return state, 0.0, False

if __name__ == "__main__":
    student = StudentSimulator(None)
    student.progress(None, None)
