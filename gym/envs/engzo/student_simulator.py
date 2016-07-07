from models import Activity, Knowledge, KnowledgeGroup
import numpy as np
import logging

logger = logging.getLogger(__name__)

STATE_TRANSFER_TYPE_PRE_UPPER_BOUNDED = "preliminary.upper.bounded.state.transfer"
STATE_TRANSFER_TYPE_PRE_FACTORED = "preliminary.factored.state.transfer"
STATE_TRANSFER_TYPE_IGNORE_PRE = "ignore.preliminary.state.transfer"
REWARD_TYPE_MASTERY_DIFF = "mastery.diff.reward"
REWARD_TYPE_OVERALL_PERF_DIFF = "overall.performance.diff.reward"
COMPLETE_TYPE_MASTERY_AVG = "mastery.average.complete"


class StudentSimulator:
    def __init__(
            self,
            state_transfer_type=STATE_TRANSFER_TYPE_IGNORE_PRE,
            reward_type=REWARD_TYPE_MASTERY_DIFF,
            complete_type=COMPLETE_TYPE_MASTERY_AVG
    ):
        self.state_transfer_type = state_transfer_type
        self.reward_type = reward_type
        self.complete_type = complete_type

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
        new_state, reward, complete = None, 0., False

        # Predict performance
        alpha = (action.psi - state).clim(min=0.)
        phi = state[action.related_knowledge_indexes]
        y = max(1 - alpha.norm / phi.norm, 0)

        # State(knowledge mastery) update
        tau = y * alpha
        if self.state_transfer_type == STATE_TRANSFER_TYPE_IGNORE_PRE:
            new_state = state + tau
        elif self.state_transfer_type == STATE_TRANSFER_TYPE_PRE_UPPER_BOUNDED:
            upper_bound = np.average(state[action.preliminary_knowledge_indexes])
            new_state = (state + tau).clip(max=upper_bound)
        else:
            # TODO TAU_TYPE_PRE_FACTORED
            raise NotImplementedError

        # Reward
        if self.reward_type == REWARD_TYPE_MASTERY_DIFF:
            reward = np.sum(new_state - state)
        else:
            raise NotImplementedError

        # Complete?
        if self.complete_type == COMPLETE_TYPE_MASTERY_AVG:
            complete = np.average(new_state) > .8
        else:
            raise NotImplementedError

        return new_state, reward, complete
