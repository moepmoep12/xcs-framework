from abc import abstractmethod, ABC
from typing import TypeVar, List
from overrides import overrides
import copy
import random
from numbers import Number

from xcs.classifier_sets import ClassifierSet
from xcs.state import State
from xcs.classifier import Classifier
from xcs.condition import Condition
from xcs.symbol import WildcardSymbol, Symbol, ISymbol
from xcs.exceptions import OutOfRangeException, WrongStrictTypeException, EmptyCollectionException

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

    def __init__(self, wild_card_probability: float):
        """
        :param wild_card_probability: Must be number in range [0.0, 1.0]
        """

        self.wildcard_probability = wild_card_probability

    @property
    def wildcard_probability(self) -> float:
        return self._wildcard_probability

    @wildcard_probability.setter
    def wildcard_probability(self, value: float):
        """
        :param value: Must be number in range [0.0, 1.0].
        :raises:
            OutOfRangeException: If wild_card_probability is not a number in range [0.0, 1.0].
        """
        if not isinstance(value, Number) or value < 0.0 or value > 1.0:
            raise OutOfRangeException(0.0, 1.0, value)

        self._wildcard_probability = value

    @overrides
    def covering_operation(self,
                           current_state: State[SymbolType],
                           available_actions: List[ActionType]) -> ClassifierSet[SymbolType, ActionType]:
        """
        Creates a matching classifier for every available action. Every element of the condition has a probability
        of turning into a Wildcard symbol.

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
                if random.random() < self._wildcard_probability:
                    condition_symbols[i] = WildcardSymbol()
                else:
                    condition_symbols[i] = Symbol(copy.deepcopy(current_state[i]))

            cl = Classifier(condition=Condition(condition_symbols), action=action)
            result.append(cl)

        return result
