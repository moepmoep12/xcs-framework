from abc import abstractmethod, ABC
from typing import List, TypeVar, Dict, Generic
from overrides import overrides
from random import shuffle, choice
from sys import float_info
from dataclasses import dataclass

from xcs.classifier_sets import MatchSet, Population
from xcs.state import State
from xcs.components.covering import ICoveringComponent
from xcs.exceptions import WrongSubTypeException

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


@dataclass
class ChosenAction(Generic[ActionType]):
    """
    Struct that groups an action and the expected payoff upon execution.
    """
    action: ActionType
    expected_reward: float


class IPerformanceComponent(ABC):
    """
    Interface. An IPerformanceComponent is responsible for generating the match set and action selection.
    """

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
    def choose_action(self, match_set: MatchSet[SymbolType, ActionType], is_explore: bool = False) -> ChosenAction:
        """
        Chooses the action to be executed.

        :param match_set: The set of classifier that match to the given state.
        :param is_explore: Whether this iteration is an explore-iteration.
        :return: The chosen action.
        """
        pass


class PerformanceComponent(IPerformanceComponent):
    """
    A basic Performance Component.
    Implementation based upon the paper 'An algorithmic description of XCS' by Butz & Wilson 2000
    (https://doi.org/10.1007/s005000100111).
    """

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
            remaining_actions = list(set([a for a in self._available_actions if a not in actions]))
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
    def choose_action(self, match_set: MatchSet[SymbolType, ActionType], is_explore: bool = False) -> ChosenAction:
        prediction_array = self._generate_prediction_array(match_set)
        if is_explore:
            action = choice(list(prediction_array.keys()))
        else:
            action = max(prediction_array, key=prediction_array.get)

        return ChosenAction(action, prediction_array[action])

    @staticmethod
    def _generate_prediction_array(match_set: MatchSet[SymbolType, ActionType]) -> Dict[ActionType, float]:
        """
        :param match_set: The classifier under consideration.
        :return: A dict where for each available action a prediction is assigned.
        """
        prediction_array: Dict[ActionType, float] = dict()
        fitness_sums: Dict[ActionType, float] = dict()
        for cl in match_set:
            if cl.action not in prediction_array:
                prediction_array[cl.action] = 0
                fitness_sums[cl.action] = 0

            prediction_array[cl.action] += cl.prediction * cl.fitness
            fitness_sums[cl.action] += cl.fitness

        for action in prediction_array.keys():
            div = fitness_sums[action] if fitness_sums[action] != 0 else float_info.epsilon
            prediction_array[action] = prediction_array[action] / div

        return prediction_array

    @property
    def covering_component(self) -> ICoveringComponent:
        """
        :return: The covering component used for covering.
        """
        return self._covering_component

    @covering_component.setter
    def covering_component(self, value: ICoveringComponent):
        """
        :param value: An object of type ICoveringComponent.
        :raises:
            WrongSubTypeException: If value is not a subtype of ICoveringComponent.
        """
        if not isinstance(value, ICoveringComponent):
            raise WrongSubTypeException(ICoveringComponent.__name__, type(value).__name__)

        self._covering_component = value
