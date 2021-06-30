from abc import abstractmethod, ABC
from typing import List, TypeVar
from overrides import overrides
from random import shuffle

from xcs.classifier_sets import MatchSet, Population
from xcs.state import State
from xcs.components.covering import ICoveringComponent

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

    def __init__(self, min_diff_actions: int,
                 covering_component: ICoveringComponent,
                 available_actions: List[ActionType]):
        """
        :param min_diff_actions: How many different actions need to be in the match set, so that covering is not
                                 necessary.
        :param covering_component: Component used for covering operation.
        :param available_actions: All available actions.
        """
        self._min_diff_actions = min_diff_actions
        self._covering_component = covering_component
        self._available_actions = available_actions

    @overrides
    def generate_match_set(self, population: Population[SymbolType, ActionType], state: State[SymbolType]) -> \
            MatchSet[SymbolType, ActionType]:

        match_set: MatchSet[SymbolType, ActionType] = MatchSet()
        for cl in population:
            if cl.condition.matches(state):
                match_set.insert_classifier(cl)

        actions = match_set.get_available_actions()

        # use covering to create new classifier
        if len(actions) < self._min_diff_actions and self._min_diff_actions > 0:
            remaining_actions = set([a for a in self._available_actions if a not in actions])
            shuffle(remaining_actions)

            for i, action in enumerate(remaining_actions):
                if i >= self._min_diff_actions - len(actions):
                    break

                covered_cl = self.covering_component.covering_operation(state, [action])
                for cl in covered_cl:
                    match_set.insert_classifier(cl)
                    population.insert_classifier(cl)

        return match_set

    @overrides
    def choose_action(self, match_set: MatchSet[SymbolType, ActionType]) -> ActionType:
        pass

    # ------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------------- PROPERTIES ------------------------------------------------ #
    # ------------------------------------------------------------------------------------------------------------- #
    @property
    def covering_component(self) -> ICoveringComponent:
        return self._covering_component

    @covering_component.setter
    def covering_component(self, value: ICoveringComponent):
        if not isinstance(value, ICoveringComponent):
            raise ValueError(f"{value} is not of type {type(ICoveringComponent)}")

        self._covering_component = value
