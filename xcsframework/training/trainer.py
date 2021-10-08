import numpy as np
import random
import sys
from typing import TypeVar, Generic
from dataclasses import dataclass

from xcsframework.xcs.algorithm import XCS
from xcsframework.xcs.state import State
from .metrics import Metric
from .environment import IEnvironment

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


@dataclass
class DataPoint(Generic[SymbolType, ActionType]):
    """
    A simple data point holding a state and the correct action to do in that state.
    """
    state: State[SymbolType]
    correct_action: ActionType


class TrainerEnvironment:
    """
    A basic trainer using an environment for training.
    """

    def optimize(self,
                 xcs: XCS[SymbolType, ActionType],
                 environment: IEnvironment,
                 training_iterations: int) -> None:
        """
        :param xcs: The XCS to be trained.
        :param environment: The environment to train upon.
        :param training_iterations: How many training iterations will be performed.
        """
        for iteration in range(training_iterations):
            state = environment.get_state()
            # switch equally between exploration & exploitation
            exploring = random.random() < 0.5
            action = xcs.run(state=state, is_explore=exploring)
            reward = environment.execute_action(action)
            xcs.reward(value=reward, is_end_of_problem=environment.is_end_of_problem())

        sys.stdout.write(f"\rIteration {iteration}/{training_iterations}")
        sys.stdout.flush()


class TrainerSingleStepDataPoints:
    """
    Trains on a single step problem on predefined data.
    """

    def optimize(self,
                 xcs: XCS[SymbolType, ActionType],
                 data_train: [DataPoint[SymbolType, ActionType]],
                 data_valid: [DataPoint[SymbolType, ActionType]],
                 validation_metrics: [Metric],
                 batch_size: int,
                 reward: float):
        """
        :param xcs: The XCS to be trained.
        :param data_train: The training data.
        :param data_valid: The validation data.
        :param validation_metrics: The metrics used for validation.
        :param batch_size: After how many data points validation will occur.
        :param reward: The reward for correct prediction.
        :return: List of validation metrics.
        """
        metrics_history = []
        num_train_examples = len(data_train)
        epochs = int(num_train_examples / batch_size)

        # Shuffle train data
        indices = np.arange(num_train_examples)
        np.random.shuffle(indices)
        data_train_shuffled = data_train[indices]
        i = 0

        for epoch in range(epochs):
            data_batch = (data_train_shuffled[i: i + batch_size])
            i += batch_size

            for data_point in data_batch:
                # switch equally between exploration & exploitation
                exploring = random.random() < 0.5
                chosen_action = xcs.run(state=data_point.state, is_explore=exploring)
                if chosen_action == data_point.correct_action:
                    rew = reward
                else:
                    rew = 0

                # give feedback. single step problem
                xcs.reward(rew, is_end_of_problem=True)

            # validate with validation data after each epoch
            if len(validation_metrics) > 0 and data_valid is not None:
                metrics_results_epoch = []
                predictions = [None] * len(data_valid)
                actuals = [d.correct_action for d in data_valid]

                for k in range(len(data_valid)):
                    predictions[k] = xcs.query(data_valid[k].state)

                for metric in validation_metrics:
                    metrics_results_epoch.append(metric.score(predictions, actuals))
                metrics_history.append(metrics_results_epoch)

                # print progress update
                sys.stdout.write(
                    '\r' + "Epoch %i / " % (epoch + 1) + str(epochs) + " Metrics: " + str(metrics_results_epoch))
                sys.stdout.flush()
            else:
                sys.stdout.write('\r' + "Epoch %i / " % (epoch + 1) + str(epochs))
                sys.stdout.flush()

        return metrics_history
