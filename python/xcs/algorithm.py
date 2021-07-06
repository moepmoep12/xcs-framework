from typing import TypeVar, Generic, List
from dataclasses import dataclass

from xcs.components.performance import ChosenAction
from xcs.state import State
from xcs.classifier_sets import Population, ActionSet
from xcs.components import *
from xcs.exceptions import *

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


class XCS(Generic[SymbolType, ActionType]):

    # TO-DO: 1.Instantiate with a list of all available actions?
    #        2.Optional initial population?
    def __init__(self,
                 population: Population[SymbolType, ActionType],
                 performance_component: IPerformanceComponent,
                 discovery_component: IDiscoveryComponent,
                 learning_component: ILearningComponent,
                 available_actions: List[ActionType]):

        self.performance_component: IPerformanceComponent = performance_component
        self.discovery_component: IDiscoveryComponent = discovery_component
        self.learning_component: ILearningComponent = learning_component
        self._population: Population[SymbolType, ActionType] = population
        self._available_actions = available_actions
        self._expects_reward: bool = False
        self._prev_state: XcsState = XcsState()
        self._discovery_threshold = 5
        self._last_discovery_run = 0
        self._iteration = 0

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

        :param state: The current state.
        :param is_explore: Whether to explore (random action) or exploit (action selection strategy).
        :return: The chosen action to execute in the current state.
        :raises:
            AssertionError: If this is called twice in a row without calling reward() in between.
        """

        assert (not self._expects_reward)

        match_set = self._performance_component.generate_match_set(population=self._population, state=state)

        chosen_action: ChosenAction = self._performance_component.choose_action(match_set=match_set,
                                                                                is_explore=is_explore)

        self._expects_reward = True

        # remember the state for updating later
        self._prev_state.action_set = ActionSet([cl for cl in match_set if cl.action == chosen_action.action])
        self._prev_state.chosen_action = chosen_action
        self._prev_state.env_state = state
        self._prev_state.is_explore = is_explore
        self._iteration += 1

        return chosen_action.action

    def reward(self, value: float, is_end_of_problem: bool = True) -> None:
        """
        Receive reward from the environment.

        :param value: The reward received (can be negative).
        :param is_end_of_problem: Whether the RL problem ends now. For single step problem this is always true.
        :raises:
            AssertionError: If run() was not called previously.
        """

        assert self._expects_reward

        self._expects_reward = False
        self.learning_component.update_set(self._prev_state.action_set, value)

        if self._prev_state.is_explore and self._prev_state.action_set is not None:
            self._discover_classifier(self._prev_state.env_state, self._prev_state.action_set, do_subsumption=True)

        if is_end_of_problem:
            self._prev_state = XcsState()

    def _discover_classifier(self, state: State[SymbolType], classifier_set, do_subsumption: bool = True):
        average_age = sum([cl.age - self._last_discovery_run if cl.age > self._last_discovery_run else 0 for cl in
                           classifier_set]) / len(classifier_set)
        if average_age >= self._discovery_threshold:
            self._last_discovery_run = self._iteration
            classifier = self.discovery_component.discover(state, classifier_set)
            for cl in classifier:
                self._population.insert_classifier(cl, do_subsumption=do_subsumption)

    @property
    def performance_component(self) -> IPerformanceComponent:
        return self._performance_component

    @performance_component.setter
    def performance_component(self, value: IPerformanceComponent):
        if not isinstance(value, IPerformanceComponent):
            raise WrongSubTypeException(IPerformanceComponent.__name__, type(value).__name__)

        self._performance_component = value

    @property
    def discovery_component(self) -> IDiscoveryComponent:
        return self._discovery_component

    @discovery_component.setter
    def discovery_component(self, value: IDiscoveryComponent):
        if not isinstance(value, IDiscoveryComponent):
            raise ValueError(f"{value} is not of type {type(IDiscoveryComponent)}")

        self._discovery_component = value

    @property
    def learning_component(self) -> ILearningComponent:
        return self._learning_component

    @learning_component.setter
    def learning_component(self, value: ILearningComponent):
        if not isinstance(value, ILearningComponent):
            raise ValueError(f"{value} is not of type {type(ILearningComponent)}")

        self._learning_component = value
