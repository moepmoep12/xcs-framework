from typing import TypeVar, Generic, List
from dataclasses import dataclass

from xcs.components.performance import ChosenAction
from xcs.state import State
from xcs.classifier_sets import Population, ActionSet
from xcs.components import *
from xcs.exceptions import *
from xcs.constants import XCSConstants

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


@dataclass
class XcsState(Generic[SymbolType, ActionType]):
    """
    Encapsulates the state of a XCS.
    """

    env_state: State[SymbolType] = None
    action_set: ActionSet[SymbolType, ActionType] = None
    chosen_action: ChosenAction[ActionType] = None
    is_explore: bool = False
    received_reward: float = 0


class XCS(Generic[SymbolType, ActionType]):
    """
    A basic component based implementation of eXtended Learning Classifier System.
    Inspiration taken from the paper 'An algorithmic description of XCS' by Butz & Wilson 2000
    (https://doi.org/10.1007/s005000100111).
    """

    def __init__(self,
                 population: Population[SymbolType, ActionType],
                 performance_component: IPerformanceComponent,
                 discovery_component: IDiscoveryComponent,
                 learning_component: ILearningComponent,
                 available_actions: List[ActionType],
                 xcs_constants: XCSConstants = XCSConstants()):
        """
        :param population: The initial population. Can be empty.
        :param performance_component: The performance component used for action selection.
        :param discovery_component: The discovery component used for rule discovery.
        :param learning_component: The learning component used for updating classifier.
        :param available_actions: Actions available during learning.
        """

        self.performance_component: IPerformanceComponent = performance_component
        self.discovery_component: IDiscoveryComponent = discovery_component
        self.learning_component: ILearningComponent = learning_component

        self._population: Population[SymbolType, ActionType] = population
        self._available_actions = available_actions
        self._expects_reward: bool = False
        self._prev_state: XcsState = XcsState()
        self._current_state: XcsState = XcsState()
        self._iteration = 0
        self._xcs_constants = xcs_constants

    def query(self, state: State[SymbolType]) -> ActionType:
        """
        Queries the best action in the given state without updating the state of the XCS.

        :param state: The current state.
        :return: The chosen action to execute in the current state.
        """
        match_set = self._performance_component.generate_match_set(population=self._population, state=state)

        chosen_action: ChosenAction = self._performance_component.choose_action(match_set=match_set,
                                                                                is_explore=False)
        return chosen_action.action

    def run(self, state: State[SymbolType], is_explore: bool = False) -> ActionType:
        """
        Performs an iteration of the XCS algorithm. Alters the state of the XCS.
        Requires to call reward() after this.

        :param state: The current state.
        :param is_explore: Whether to explore (random action) or exploit (action selection strategy).
        :return: The chosen action to execute in the current state.
        :raises:
            AssertionError: If this is called twice in a row without calling reward() in between.
        """

        assert (not self._expects_reward)

        if 0 < self._xcs_constants.max_iterations < self._iteration:
            return

        match_set = self._performance_component.generate_match_set(population=self._population, state=state)

        chosen_action: ChosenAction = self._performance_component.choose_action(match_set=match_set,
                                                                                is_explore=is_explore)

        self._expects_reward = True
        self._iteration += 1

        self._current_state = XcsState(env_state=state,
                                       action_set=ActionSet(
                                           [cl for cl in match_set if cl.action == chosen_action.action]),
                                       chosen_action=chosen_action,
                                       is_explore=is_explore,
                                       received_reward=0)

        return chosen_action.action

    def reward(self, value: float, is_end_of_problem: bool = True) -> None:
        """
        Receive reward from the environment. Requires to have called run() beforehand.

        :param value: The reward received (can be negative).
        :param is_end_of_problem: Whether the RL problem ends now. For single step problem this is always true.
        :raises:
            AssertionError: If run() was not called previously.
        """

        assert self._expects_reward

        self._expects_reward = False

        reward = value
        set_to_update = self._current_state.action_set
        is_explore = self._current_state.is_explore
        state = self._current_state.env_state

        if self._prev_state.action_set is not None:
            reward = self._prev_state.received_reward + self._xcs_constants.gamma * value
            set_to_update = self._prev_state.action_set
            is_explore = self._prev_state.is_explore
            state = self._prev_state.env_state

        self.learning_component.update_set(set_to_update, reward)

        if self._xcs_constants.do_learning_subsumption:
            self._do_action_set_subsumption(set_to_update)

        if is_explore:
            discovered_classifier = self.discovery_component.discover(self._iteration, state, set_to_update)
            for cl in discovered_classifier:
                self._population.insert_classifier(cl, do_subsumption=self._xcs_constants.do_discovery_subsumption)

        if is_end_of_problem:
            self._prev_state = XcsState()
            self._current_state = XcsState()
        else:
            self._prev_state = self._current_state
            self._prev_state.received_reward = value

    def reset(self):
        """
        Resets the state of this XCS. Population will be cleared.
        """
        self._prev_state = XcsState()
        self._current_state = XcsState()
        self._iteration = 0
        self._population.trim_population(0)
        self._expects_reward = False

    # todo: fix bug where a classifier to be removed is not in the population
    def _do_action_set_subsumption(self, action_set: ActionSet[SymbolType, ActionType]):
        if len(action_set) <= 1:
            return

        most_general_classifier = None
        for cl in action_set:
            if self.population.subsumption_criteria.can_subsume(cl):
                if most_general_classifier is None or cl.subsumes(most_general_classifier):
                    most_general_classifier = cl

        classifier_to_remove = []

        if most_general_classifier is not None:
            for cl in action_set:
                if most_general_classifier.subsumes(cl):
                    most_general_classifier.numerosity += cl.numerosity
                    classifier_to_remove.append(cl)

            for cl in classifier_to_remove:
                self.population.remove_classifier(cl)
                action_set.remove_classifier(cl)

    @property
    def population(self) -> Population[SymbolType, ActionType]:
        """
        :return: The population of a XCS represents the knowledge base.
        """
        return self._population

    @property
    def performance_component(self) -> IPerformanceComponent:
        """
        :return: The performance component used for action selection.
        """
        return self._performance_component

    @performance_component.setter
    def performance_component(self, value: IPerformanceComponent):
        """
        :param value: An object of type IPerformanceComponent.
        :raises:
            WrongSubTypeException: If value is not a subtype of IPerformanceComponent.
        """
        if not isinstance(value, IPerformanceComponent):
            raise WrongSubTypeException(IPerformanceComponent.__name__, type(value).__name__)

        self._performance_component = value

    @property
    def discovery_component(self) -> IDiscoveryComponent:
        """
        :return: The discovery component used for rule discovery.
        """
        return self._discovery_component

    @discovery_component.setter
    def discovery_component(self, value: IDiscoveryComponent):
        """
        :param value: An object of type IDiscoveryComponent.
        :raises:
            WrongSubTypeException: If value is not a subtype of IDiscoveryComponent.
        """
        if not isinstance(value, IDiscoveryComponent):
            raise ValueError(f"{value} is not of type {type(IDiscoveryComponent)}")

        self._discovery_component = value

    @property
    def learning_component(self) -> ILearningComponent:
        """
        :return: The learning component used for updating classifier.
        """
        return self._learning_component

    @learning_component.setter
    def learning_component(self, value: ILearningComponent):
        """
        :param value: An object of type ILearningComponent.
        :raises:
            WrongSubTypeException: If value is not a subtype of ILearningComponent.
        """
        if not isinstance(value, ILearningComponent):
            raise ValueError(f"{value} is not of type {type(ILearningComponent)}")

        self._learning_component = value
