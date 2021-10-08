import random
from overrides import overrides
from typing import Collection

from xcs_framework.xcs.components.discovery import GeneticAlgorithm, SymbolType, ActionType
from xcs_framework.xcs.selection import IClassifierSelectionStrategy, RouletteWheelSelection
from xcs_framework.xcs.classifier import Classifier
from xcs_framework.xcs.state import State

from xcs_framework.xcsr.constants import XCSRGAConstants


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
    def _mutate(self, classifier: Classifier[SymbolType, ActionType], state: State[SymbolType]):
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
