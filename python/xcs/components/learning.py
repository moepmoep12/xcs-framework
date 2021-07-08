from abc import abstractmethod, ABC
from overrides import overrides
from typing import TypeVar
from math import inf
from numbers import Number

from xcs.classifier_sets import ClassifierSet
from xcs.exceptions import OutOfRangeException

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class ILearningComponent(ABC):

    # todo: docstring
    @abstractmethod
    def update_set(self, classifier_set: ClassifierSet[SymbolType, ActionType], reward: float):
        pass


class QLearningBasedComponent(ILearningComponent):

    # todo: 'settings/param'-object as argument?
    def __init__(self, learning_rate_prediction: float):
        """
        :param learning_rate_prediction: The learning rate >0 for updating the prediction of a classifier.
        """
        self.learning_rate_prediction = learning_rate_prediction
        self._epsilon_zero = 0
        self._learning_rate_fitness = 0
        self._accuracy_power = 5

    @overrides
    def update_set(self, classifier_set: ClassifierSet[SymbolType, ActionType], reward: float):
        for cl in classifier_set:
            cl.increment_experience()

            if cl.experience < (1.0 / self.learning_rate_prediction):
                cl.epsilon += (abs(reward - cl.prediction) - cl.epsilon) / cl.experience
                cl.prediction += (reward - cl.prediction) / cl.experience
                cl.action_set_size += (classifier_set.numerosity_sum() - cl.action_set_size) / cl.experience
            else:
                cl.epsilon += self.learning_rate_prediction * (abs(reward - cl.prediction) - cl.epsilon)
                cl.prediction += self.learning_rate_prediction * (reward - cl.prediction)
                cl.action_set_size += self.learning_rate_prediction * (
                        classifier_set.numerosity_sum() - cl.action_set_size)

        self._update_fitness(classifier_set)

        # todo: subsumption

    def _update_fitness(self, classifier_set: ClassifierSet[SymbolType, ActionType]):
        """
        Updates the fitness of each classifier.

        :param classifier_set: The set of classifier that will be updated.
        """
        accuracy_sum: float = 0
        accuracy_dict = dict()
        for cl in classifier_set:
            accuracy = self._classifier_accuracy(cl)
            accuracy_dict[cl] = accuracy
            accuracy_sum += accuracy * cl.numerosity

        for cl in classifier_set:
            cl.fitness += self.learning_rate_prediction * (
                    (accuracy_dict[cl] * cl.numerosity) / accuracy_sum - cl.fitness)

    # todo: docstring
    def _classifier_accuracy(self, classifier) -> float:
        if classifier.experience == 0:
            return 0.0
        return 1.0 if classifier.epsilon <= self._epsilon_zero else self._learning_rate_fitness * (
                (classifier.epsilon / self._epsilon_zero) ** -self._accuracy_power)

    @property
    def learning_rate_prediction(self) -> float:
        """
        :return: The learning rate in range ]0.0, inf] for updating the prediction of a classifier.
        """
        return self._learning_rate_prediction

    @learning_rate_prediction.setter
    def learning_rate_prediction(self, value: float):
        """
        :param value: Float in range ]0.0, inf].
        : raises:
            OutOfRangeException: If value is not a float in range ]0.0, inf].
        """
        if not isinstance(value, Number) or value <= 0.0:
            raise OutOfRangeException(0.0, inf, value)

        self._learning_rate_prediction = value
