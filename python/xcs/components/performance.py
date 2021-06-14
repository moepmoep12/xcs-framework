from abc import abstractmethod, ABC
from typing import List, TypeVar
from overrides import overrides

from xcs.classifier_sets import MatchSet, Population
from xcs.state import State

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class IPerformanceComponent(ABC):

    @abstractmethod
    def generate_match_set(self, population: Population[SymbolType, ActionType], state: State[SymbolType]) -> \
            MatchSet[SymbolType, ActionType]:
        """
        Generates a match set to given population in a specific state. A match set consists of all classifiers
        that match to the given situation.
        :param population: The set of classifiers to be considered.
        :param state: The state to check against.
        :return: All classifiers from the population that match to the situation.
        """
        pass

    @abstractmethod
    def choose_action(self, match_set: MatchSet[SymbolType, ActionType]) -> ActionType:
        pass


class PerformanceComponent(IPerformanceComponent):

    @overrides
    def generate_match_set(self, population: Population[SymbolType, ActionType], state: State[SymbolType]) -> \
            MatchSet[SymbolType, ActionType]:

        match_set: MatchSet[SymbolType, ActionType] = MatchSet()
        for cl in population:
            if cl.condition.matches(state):
                match_set.append(cl)

        return match_set

    @overrides
    def choose_action(self, match_set: MatchSet[SymbolType, ActionType]) -> ActionType:
        pass
