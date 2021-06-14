from abc import abstractmethod, ABC
from typing import List, TypeVar, Callable
from overrides import overrides
import random

from xcs.classifier_sets import ClassifierSet
from xcs.classifier import Classifier

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')

# The signature for a function returning the score of a classifier
score_function_type = Callable[[Classifier[SymbolType, ActionType]], float]


class IClassifierSelectionStrategy(ABC):
    """
    Interface. Responsible for classifier selection.
    """

    @abstractmethod
    def select_classifier(self,
                          classifier_set: ClassifierSet[SymbolType, ActionType],
                          score_function: score_function_type) -> int:
        """
        Chooses a single classifier from the given set.
        :param classifier_set: The set to choose from.
        :param score_function: A function that returns the score of a classifier.
        :return: The index of chosen classifier.
        """
        pass


class TournamentSelection(IClassifierSelectionStrategy):

    def __init__(self, tournament_size: int):
        self._tournament_size = tournament_size

    @overrides
    def select_classifier(self,
                          classifier_set: ClassifierSet[SymbolType, ActionType],
                          score_function: score_function_type) -> int:

        best: Classifier[SymbolType, ActionType] = None
        best_index = 0
        indices = [i for i in range(len(classifier_set))]

        # select competing individuals
        for _ in range(self._tournament_size):
            index = indices[random.randint(0, len(indices) - 1)]
            if best is None or score_function(classifier_set[index]) > score_function(best):
                best = classifier_set[index]
                best_index = index
            indices.remove(index)

        return best_index


class RouletteWheelSelection(IClassifierSelectionStrategy):

    @overrides
    def select_classifier(self,
                          classifier_set: ClassifierSet[SymbolType, ActionType],
                          score_function: score_function_type) -> int:

        score_sum = sum([score_function(cl) for cl in classifier_set])
        choice_point = random.random() * score_sum
        score_sum = 0
        for i, cl in enumerate(classifier_set):
            score_sum += score_function(cl)
            if score_sum > choice_point:
                return i
