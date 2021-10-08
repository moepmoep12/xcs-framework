from abc import abstractmethod, ABC
from typing import TypeVar, List
from overrides import overrides
import copy
import random

from xcsframework.xcs.classifier_sets import ClassifierSet
from xcsframework.xcs.state import State
from xcsframework.xcs.classifier import Classifier
from xcsframework.xcs.condition import Condition
from xcsframework.xcs.symbol import WildcardSymbol, ISymbol, Symbol
from xcsframework.xcs.exceptions import EmptyCollectionException
from xcsframework.xcs.constants import CoveringConstants

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class ICoveringComponent(ABC):
    """
    Interface. An ICoveringComponent is responsible for creating new classifier in a given situation,
    without exploiting the current knowledge base.
    The operation is usually called if no classifier match to a given state.
    """

    @abstractmethod
    def covering_operation(self,
                           current_state: State[SymbolType],
                           available_actions: List[ActionType]) -> ClassifierSet[SymbolType, ActionType]:
        """
        Creates new classifier from the given state.

        :param current_state: The current state.
        :param available_actions: All available actions to choose from.
        :return: A set of new classifiers.
        """
        pass


class CoveringComponent(ICoveringComponent):
    """
    Basic CoveringComponent.
    """

    def __init__(self, covering_constants: CoveringConstants = CoveringConstants()):
        """
        :param covering_constants: Constants used in this component.
        """
        self._covering_constants: CoveringConstants = covering_constants

    @overrides
    def covering_operation(self,
                           current_state: State[SymbolType],
                           available_actions: List[ActionType]) -> ClassifierSet[SymbolType, ActionType]:
        """
        Creates a matching classifier for each available action.
        Every element of the condition has a probability to become a Wildcard symbol.

        :param current_state: The current state.
        :param available_actions: All available actions to choose from.
        :return: A set of new classifiers.
        :raises:
            EmptyCollectionException: If current_state or available_actions are empty.
        """

        if len(current_state) == 0:
            raise EmptyCollectionException('current_state')
        if len(available_actions) == 0:
            raise EmptyCollectionException('current_state')

        result = ClassifierSet()

        for action in available_actions:
            condition_symbols: List[ISymbol[SymbolType]] = [None] * len(current_state)

            for i in range(len(condition_symbols)):
                if random.random() < self.covering_constants.wildcard_probability:
                    condition_symbols[i] = WildcardSymbol()
                else:
                    condition_symbols[i] = self._create_symbol(current_state[i])

            cl = Classifier(condition=Condition(condition_symbols), action=action)
            result.insert_classifier(cl)

        return result

    @property
    def covering_constants(self) -> CoveringConstants:
        """
        :return: Covering constants used in this component.
        """
        return self._covering_constants

    def _create_symbol(self, value: SymbolType) -> ISymbol[SymbolType]:
        """
        Factory method for symbol creation.
        :param value: The value for the symbol. Can be a ref!
        :return: Newly created symbol.
        """
        # because value can be a ref, we make a deep copy
        return Symbol(copy.deepcopy(value))
