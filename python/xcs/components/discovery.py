from abc import abstractmethod, ABC
from overrides import overrides
from typing import TypeVar, List
import copy
import random

from xcs.classifier import Classifier
from xcs.classifier_sets import Population, ClassifierSet
from xcs.symbol import WildcardSymbol
from xcs.condition import Condition
from xcs.state import State
from xcs.selection import IClassifierSelectionStrategy

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class IDiscoveryComponent(ABC):

    @abstractmethod
    def discover(self, state: State[SymbolType], classifier_set: ClassifierSet[SymbolType, ActionType]):
        """
        Discovers new classifier based on existing classifier in a specific state.
        :param state: The current state.
        :param classifier_set: The set used for generating new classifier (== current knowledge base).
        """
        pass


class GeneticAlgorithm(IDiscoveryComponent):

    # TO-DO: Add constructor arguments
    def __init__(self):
        self._mutation_rate = 0
        self._mutate_action = False
        self._available_actions: List[ActionType] = []
        self._selection_strategy: IClassifierSelectionStrategy = None
        self._fitness_reduction: float = 0.1

    @overrides
    def discover(self, state: State[SymbolType], classifier_set: ClassifierSet[SymbolType, ActionType]):

        parent1 = self._choose_parent(classifier_set)
        parent2 = self._choose_parent(classifier_set)

        child1 = self._generate_child(parent1)
        child2 = self._generate_child(parent2)

        self._crossover(child1, child2)
        self._mutate(child1, state)
        self._mutate(child2, state)

        return ClassifierSet([child1, child2])

    @staticmethod
    def _generate_child(parent: Classifier[SymbolType, ActionType]) -> Classifier[SymbolType, ActionType]:
        """
        Generates a child classifier from a parent. Uses deep copy.
        """
        child = Classifier(copy.deepcopy(parent.condition), copy.deepcopy(parent.action))
        child.fitness = parent.fitness / parent.numerosity
        child.prediction = parent.prediction
        child.epsilon = parent.epsilon
        return child

    def _choose_parent(self, classifier_set: ClassifierSet[SymbolType, ActionType]) \
            -> Classifier[SymbolType, ActionType]:
        """
        Chooses a classifier as parent with the given SelectionStrategy and the fitness as criteria.
        """
        return classifier_set[self._selection_strategy.select_classifier(classifier_set, lambda cl: cl.fitness)]

    def _mutate(self, classifier: Classifier[SymbolType, ActionType], state: State[SymbolType]):
        """
        Mutates a classifier by changing some of its condition symbols and the action if enabled.
        """
        for i in range(len(classifier.condition)):
            if random.random() < self._mutation_rate:
                if isinstance(classifier.condition[i], WildcardSymbol):
                    classifier.condition[i] = state[i]
                else:
                    classifier.condition[i] = WildcardSymbol()

        if self._mutate_action:
            actions = set(self._available_actions)
            actions.remove(classifier.action)
            if len(actions) > 0:
                classifier.action = actions[random.randint(0, len(actions) - 1)]

    def _crossover(self, cl1: Classifier[SymbolType, ActionType], cl2: Classifier[SymbolType, ActionType]):
        """
        Performs the crossover operation on two classifier.

        """
        performed_crossover = False

        # TO-DO: Perform crossover according to strategy

        if performed_crossover:
            cl1.prediction = cl2.prediction = (cl1.prediction + cl2.prediction) / 2
            cl1.epsilon = cl2.epsilon = (cl1.epsilon + cl2.epsilon) / 2
            cl1.fitness = cl2.fitness = (cl1.fitness + cl2.fitness) / 2

        cl1.fitness *= self._fitness_reduction
        cl2.fitness *= self._fitness_reduction

    def _uniform_crossover(self,
                           cl1: Classifier[SymbolType, ActionType],
                           cl2: Classifier[SymbolType, ActionType]) -> bool:

        performed_crossover = False

        for i in range(len(cl1.condition)):
            if random.random() >= 0.5:
                self._swap_symbols(cl1.condition, cl2.condition, i, i)
                performed_crossover = True

        return performed_crossover

    def _one_point_crossover(self,
                             cl1: Classifier[SymbolType, ActionType],
                             cl2: Classifier[SymbolType, ActionType]) -> bool:

        from_index = random.randint(0, len(cl1.condition) - 1)
        return self._swap_symbols(cl1.condition, cl2.condition, from_index, len(cl1.condition))

    def _two_point_crossover(self,
                             cl1: Classifier[SymbolType, ActionType],
                             cl2: Classifier[SymbolType, ActionType]) -> bool:

        from_index = random.randint(0, len(cl1.condition) - 1)
        to_index = random.randint(0, len(cl1.condition) - 1)

        if from_index > to_index:
            from_index, to_index = to_index, from_index

        return self._swap_symbols(cl1.condition, cl2.condition, from_index, to_index)

    @staticmethod
    def _swap_symbols(condition1: Condition[SymbolType], condition2: Condition[SymbolType],
                      from_index: int, to_index: int) -> bool:
        """
        Swaps the symbols of two classifier.
        :param from_index: Starting index (inclusive).
        :param to_index: End index (inclusive).
        :return: Whether anything was swapped.
        """
        if condition1 is None or condition2 is None:
            raise ValueError(f"Tried swapping a condition that is None")
        if len(condition1) == 0:
            raise ValueError(f"Cannot swap a empty condition {condition1}")
        if len(condition2) == 0:
            raise ValueError(f"Cannot swap a empty condition {condition2}")
        if len(condition1) != len(condition2):
            raise ValueError(f"Condition length of {len(condition1)} != {len(condition2)}")
        if condition1 is condition2:
            raise ValueError(f"Tried swapping cond1:{condition1} with itself cond2:{condition2}")
        if from_index < 0 or from_index >= len(condition1):
            raise ValueError(f"from_index {from_index} is out of range [0,{len(condition1) - 1}]")
        if to_index < 0 or to_index >= len(condition1):
            raise ValueError(f"to_index {to_index} is out of range [0,{len(condition1) - 1}]")
        if from_index > to_index:
            raise ValueError(f"from_index {from_index} > {to_index} to_index")

        swapped = False

        for i in range(from_index, to_index + 1):
            condition1[i], condition2[i] = condition2[i], condition1[i]
            swapped = True

        return swapped
