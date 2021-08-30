from abc import abstractmethod, ABC
from overrides import overrides
from typing import TypeVar

from xcs.classifier_sets import ClassifierSet
from xcs.constants import LearningConstants, FitnessConstants

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class ILearningComponent(ABC):
    """
    Interface. An ILearningComponent is responsible for updating the attributes of classifier according
    to the received reward.
    """

    @abstractmethod
    def update_set(self, classifier_set: ClassifierSet[SymbolType, ActionType], reward: float):
        """
        Updates the given set of classifier with the received reward.

        :param classifier_set: The classifier to be updated.
        :param reward: The received reward.
        """

        pass


class QLearningBasedComponent(ILearningComponent):
    """
    Basic Learning Component.
    Implementation based upon the paper 'An algorithmic description of XCS' by Butz & Wilson 2000
    (https://doi.org/10.1007/s005000100111).
    """

    def __init__(self,
                 learning_constants: LearningConstants = LearningConstants(),
                 fitness_constants: FitnessConstants = FitnessConstants()):
        """
        :param learning_constants: Constants used for learning (prediction, error) updates.
        :param fitness_constants: Constants used for fitness updates.
        """
        self._learning_constants = learning_constants
        self._fitness_constants = fitness_constants

    @overrides
    def update_set(self, classifier_set: ClassifierSet[SymbolType, ActionType], reward: float):
        for cl in classifier_set:
            cl.increment_experience()

            if cl.experience < (1.0 / self.learning_constants.beta):
                cl.epsilon += (abs(reward - cl.prediction) - cl.epsilon) / cl.experience
                cl.prediction += (reward - cl.prediction) / cl.experience
                cl.action_set_size += (classifier_set.numerosity_sum() - cl.action_set_size) / cl.experience
            else:
                cl.epsilon += self.learning_constants.beta * (abs(reward - cl.prediction) - cl.epsilon)
                cl.prediction += self.learning_constants.beta * (reward - cl.prediction)
                cl.action_set_size += self.learning_constants.beta * (
                        classifier_set.numerosity_sum() - cl.action_set_size)

        self._update_fitness(classifier_set)

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
            cl.fitness += self.learning_constants.beta * (
                    (accuracy_dict[cl] * cl.numerosity) / accuracy_sum - cl.fitness)

    def _classifier_accuracy(self, classifier) -> float:
        """
        :param classifier: The classifier the accuracy will be calculated for.
        :return: How accurate a classifier predicts the expected reward in range [0,1].
        """
        if classifier.experience == 0:
            return 0.0
        if classifier.epsilon <= self.learning_constants.epsilon_zero:
            return 1.0
        else:
            return self.fitness_constants.alpha * (
                    (classifier.epsilon / self.learning_constants.epsilon_zero) ** -self.fitness_constants.nu)

    @property
    def learning_constants(self) -> LearningConstants:
        """
        :return: Constants used for learning updates.
        """
        return self._learning_constants

    @property
    def fitness_constants(self) -> FitnessConstants:
        """
        :return: Constants used for fitness updates.
        """
        return self._fitness_constants
