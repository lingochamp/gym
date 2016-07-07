from models import Activity, Knowledge, KnowledgeGroup
import numpy as np
import logging

logger = logging.getLogger(__name__)

TAU_TYPE_PRE_UPPER_BOUNDED = "preliminary.upper.bounded.tau"
TAU_TYPE_PRE_FACTORED = "preliminary.factored.tau"
TAU_TYPE_IGNORE_PRE = "ignore.preliminary.tau"
REWARD_TYPE_TAU = "tau.reward"
REWARD_TYPE_OVERALL_PERF = "overall.performance.reward"


class StudentSimulator:
    def __init__(self, tau_type=TAU_TYPE_IGNORE_PRE, reward_type=REWARD_TYPE_TAU):
        self.tau_type = tau_type
        self.reward_type = REWARD_TYPE_TAU

    def progress(self, state, action):
        """

        Args:
            state (:obj:`np.ndarray` of float): Knowledge mastery of student.
            action (:obj:`Activity`): Activity taken by student.

        Returns:
            new_state (:obj:`np.ndarray` of float): Updated knowledge mastery.
            reward (float): Amount of reward achieved by the previous action.
            complete (bool): If student meet the criteria of completion.

        """
        assert isinstance(action, Activity)
        assert state.shape == action.psi.shape
        tau, reward, complete = np.zeros(state.shape), 0., False

        # Predict performance
        alpha = (action.psi - state).clim(min=0.)
        phi = state[action.related_knowledge_indexes]
        y = max(1 - alpha.norm / phi.norm, 0)

        # State(knowledge mastery) update
        if self.tau_type == TAU_TYPE_IGNORE_PRE:
            tau = y * alpha
        else:
            # TODO
            raise NotImplementedError

        # Reward
        if self.reward_type == REWARD_TYPE_TAU:
            reward = np.sum(tau)

        return state + tau, reward, complete
