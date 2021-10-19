import random
from overrides import overrides
from typing import Collection
from numbers import Number

from xcsframework.xcs.components.discovery import GeneticAlgorithm, SymbolType, ActionType
from xcsframework.xcs.selection import IClassifierSelectionStrategy, RouletteWheelSelection
from xcsframework.xcs.classifier import Classifier
from xcsframework.xcs.condition import Condition
from xcsframework.xcs.state import State
from xcsframework.xcs.exceptions import NoneValueException, OutOfRangeException, EmptyCollectionException

from xcsframework.xcsr.constants import XCSRGAConstants


class OBGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self,
                 available_actions: Collection[ActionType],
                 selection_strategy: IClassifierSelectionStrategy = RouletteWheelSelection(),
                 ga_constants: XCSRGAConstants = XCSRGAConstants()):
        """
        :param selection_strategy: The strategy used for selecting parent classifier.
        :param available_actions: Available actions to choose from when mutating the action of a classifier.
        :param ga_constants: Constants used in this ga.
        :raises:
            EmptyCollectionException: If available_actions is empty.
        """
        super(OBGeneticAlgorithm, self).__init__(available_actions=available_actions,
                                                 selection_strategy=selection_strategy,
                                                 ga_constants=ga_constants)

    @overrides
    def _mutate(self, classifier: Classifier[Number, ActionType], state: State[Number]):
        """
        Mutates a classifier by changing some of its condition symbols and the action if enabled.
        """
        for i in range(len(classifier.condition)):
            if random.random() < self.ga_constants.mutation_rate:
                classifier.condition[i]._lower += random.uniform(- self.ga_constants.max_mutation_change,
                                                                 self.ga_constants.max_mutation_change)
                # keep it in range
                if self.ga_constants.truncate_to_range:
                    classifier.condition[i]._lower = min(
                        max(self.ga_constants.min_value, classifier.condition[i].lower_value),
                        self.ga_constants.max_value)

                classifier.condition[i]._upper += random.uniform(- self.ga_constants.max_mutation_change,
                                                                 self.ga_constants.max_mutation_change)
                # keep it in range
                if self.ga_constants.truncate_to_range:
                    classifier.condition[i]._upper = min(
                        max(self.ga_constants.min_value, classifier.condition[i].upper_value),
                        self.ga_constants.max_value)

                # swap to keep order
                if classifier.condition[i].lower_value > classifier.condition[i].upper_value:
                    tmp = classifier.condition[i].lower_value
                    classifier.condition[i]._lower = classifier.condition[i].upper_value
                    classifier.condition[i]._upper = tmp

        if self.ga_constants.mutate_action:
            actions = list(set(self._available_actions))
            actions.remove(classifier.action)
            if len(actions) > 0:
                classifier._action = actions[random.randint(0, len(actions) - 1)]

    @staticmethod
    def restore_order_of_bounds(symbol):
        symbol._lower, symbol._upper = symbol._upper, symbol._lower

    @staticmethod
    def _swap_symbols(condition1: Condition[SymbolType], condition2: Condition[SymbolType],
                      from_index: int, to_index: int) -> bool:
        """
        Swaps the symbols of two classifier.

        :param from_index: Starting index (inclusive).
        :param to_index: End index (inclusive).
        :return: Whether anything was swapped.
        :raises:
            NoneValueException: If any required argument is None.
            EmptyCollectionException: If any condition is empty.
            OutOfRangeException: If from_index or to_index is not in range [0, len(condition1) -1]
            ValueError: If condition1 == condition2 or the conditions have different length.
        """
        if condition1 is None or condition2 is None:
            raise NoneValueException('condition1/2')
        if len(condition1) == 0:
            raise EmptyCollectionException('condition1')
        if len(condition2) == 0:
            raise EmptyCollectionException('condition2')
        if len(condition1) != len(condition2):
            raise ValueError(f"Condition length of {len(condition1)} != {len(condition2)}")
        if condition1 is condition2:
            raise ValueError(f"Tried swapping cond1:{condition1} with itself cond2:{condition2}")
        if from_index < 0 or from_index >= len(condition1):
            raise OutOfRangeException(0, len(condition1) - 1, from_index)
        if to_index < 0 or to_index >= len(condition1):
            raise OutOfRangeException(0, len(condition1) - 1, to_index)
        if from_index > to_index:
            raise ValueError(f"from_index {from_index} > {to_index} to_index")

        swapped = False

        for i in range(from_index, to_index + 1):
            swapped_allele = False

            if random.random() >= 0.5:
                condition1[i]._lower, condition2[i]._lower = condition2[i]._lower, condition1[i]._lower
                swapped = True
                swapped_allele = True

            if random.random() >= 0.5:
                condition1[i]._upper, condition2[i]._upper = condition2[i]._upper, condition1[i]._upper
                swapped = True
                swapped_allele = True

            if swapped_allele and condition1[i].lower_value > condition1[i].upper_value:
                OBGeneticAlgorithm.restore_order_of_bounds(condition1[i])

            if swapped_allele and condition2[i].lower_value > condition2[i].upper_value:
                OBGeneticAlgorithm.restore_order_of_bounds(condition2[i])

        return swapped
