import random
import sys
from typing import TypeVar

from xcsframework.xcs.algorithm import XCS
from xcsframework.training.environment import IEnvironment

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class TrainerEnvironment:
    """
    A basic trainer using an environment for training.
    """

    def optimize(self,
                 xcs: XCS[SymbolType, ActionType],
                 environment: IEnvironment,
                 training_iterations: int,
                 explore_probability=0.5) -> None:
        """
        :param xcs: The XCS to be trained.
        :param environment: The environment to train upon.
        :param training_iterations: How many training iterations will be performed.
        """
        reward_history = []
        reward_epoch = []
        for iteration in range(training_iterations):
            state = environment.get_state()
            # switch equally between exploration & exploitation
            exploring = random.random() < explore_probability
            action = xcs.run(state=state, is_explore=exploring)
            reward = environment.execute_action(action)
            reward_epoch.append(reward)
            end_of_problem = environment.is_end_of_problem()
            xcs.reward(value=reward, is_end_of_problem=end_of_problem)
            if end_of_problem:
                reward_history.append(reward_epoch)
                reward_epoch = []

            # sys.stdout.write(f"\rIteration {iteration + 1}/{training_iterations}")
            # sys.stdout.flush()

        return reward_history
