from abc import abstractmethod, ABC
from typing import TypeVar, List
from overrides import overrides
import copy
import random

from xcs.classifier_sets import ClassifierSet
from xcs.state import State
from xcs.classifier import Classifier
from xcs.condition import Condition
from xcs.symbol import WildcardSymbol, Symbol, ISymbol

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class ICoveringComponent(ABC):
    """
    Interface. An ICoveringComponent is responsible for creating new classifier in a given situation.
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
        :param wild_card_probability: Must be float between in range [0.0, 1.0]
        """
        if not isinstance(wild_card_probability, float):
            raise ValueError(f"{wild_card_probability} is not of type float")
        if wild_card_probability < 0.0 or wild_card_probability > 1.0:
            raise ValueError(f"{wild_card_probability} out of range [0.0, 1.0]")

        self._wildcard_probability: float = wild_card_probability

    @property
    def wildcard_probability(self) -> float:
        return self._wildcard_probability

    @wildcard_probability.setter
    def wildcard_probability(self, value: float):
        """
        :param value: Must be float between in range [0.0, 1.0]
        """
        if not isinstance(value, float):
            raise ValueError(f"{value} is not of type float")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{value} out of range [0.0, 1.0]")

        self._wildcard_probability = value

    @overrides
    def covering_operation(self,
                           current_state: State[SymbolType],
                           available_actions: List[ActionType]) -> ClassifierSet[SymbolType, ActionType]:
        """
        Creates a matching classifier for every available action. Every element of the condition has a probability
        of turning into a Wildcard symbol.
        """

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
