from typing import TypeVar, Generic

from xcs.state import State
from xcs.classifier_sets import Population
from xcs.components import *
from xcs.exceptions import *

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class XCS(Generic[SymbolType, ActionType]):

    # TO-DO: 1.Instantiate with a list of all available actions?
    #        2.Optional initial population?
    def __init__(self,
                 performance_component: IPerformanceComponent,
                 discovery_component: IDiscoveryComponent,
                 learning_component: ILearningComponent,
                 covering_component: ICoveringComponent):

        self.performance_component: IPerformanceComponent = performance_component
        self.discovery_component: IDiscoveryComponent = discovery_component
        self.learning_component: ILearningComponent = learning_component
        self.covering_component: ICoveringComponent = covering_component
        self._population: Population[SymbolType, ActionType] = None

    def explore(self, state: State[SymbolType, ActionType]) -> ActionType:
        """
        Performs an iteration of the XCS algorithm in explore mode.
        Exploring means discovering new behaviour instead of exploiting the current knowledge.
        :param state: The current state.
        :return: Returns the chosen action.
        """
        match_set = self._performance_component.generate_match_set(self._population, state)
        if len(match_set) == 0:
            # TO-DO: passing available actions
            match_set.extend(self.covering_component.covering_operation(state, []))

        action: ActionType = self._performance_component.choose_action(match_set)

        pass

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

    @property
    def covering_component(self) -> ICoveringComponent:
        return self._covering_component

    @covering_component.setter
    def covering_component(self, value: ICoveringComponent):
        if not isinstance(value, ICoveringComponent):
            raise ValueError(f"{value} is not of type {type(ICoveringComponent)}")

        self._covering_component = value
