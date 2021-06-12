from abc import abstractmethod, ABC
from typing import List
from overrides import overrides

from xcs.classifier_sets import SymbolType, ActionType, ActionSet, MatchSet, Population


class IPerformanceComponent(ABC):

    @abstractmethod
    def generate_match_set(self, population: Population[SymbolType, ActionType], situation: List[SymbolType]) -> \
            MatchSet[SymbolType, ActionType]:
        pass

    @abstractmethod
    def choose_action(self, match_set: MatchSet[SymbolType, ActionType]) -> ActionType:
        pass


class PerformanceComponent(IPerformanceComponent):

    @overrides
    def generate_match_set(self, population: Population[SymbolType, ActionType], situation: List[SymbolType]) -> \
            MatchSet[SymbolType, ActionType]:

        match_set: MatchSet[SymbolType, ActionType] = MatchSet()
        for cl in population:
            if cl.condition.matches(situation):
                match_set.append(cl)

        return match_set

    @overrides
    def choose_action(self, match_set: MatchSet[SymbolType, ActionType]) -> ActionType:
        pass
