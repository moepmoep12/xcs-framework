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

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class IDiscoveryComponent(ABC):

    @abstractmethod
    def discover(self, population: Population[SymbolType, ActionType], state: State[SymbolType],
                 classifier_set: ClassifierSet[SymbolType, ActionType]) -> ClassifierSet[SymbolType, ActionType]:
        pass


class GeneticAlgorithm(IDiscoveryComponent):

    def __init__(self):
        self._mutation_rate = 0
        self._mutate_action = False
        self._available_actions: List[ActionType] = []

    @overrides
    def discover(self, population: Population[SymbolType, ActionType], state: State[SymbolType],
                 classifier_set: ClassifierSet[SymbolType, ActionType]) -> ClassifierSet[SymbolType, ActionType]:

        parent1 = self.choose_parent(population)
        parent2 = self.choose_parent(population)

        child1 = self._generate_child(parent1)
        child2 = self._generate_child(parent2)

        self._crossover(child1, child2)
        self._mutate(child1, state)
        self._mutate(child2, state)

        return ClassifierSet([child1, child2])

    @staticmethod
    def _generate_child(parent: Classifier[SymbolType, ActionType]) -> Classifier[SymbolType, ActionType]:
        child = copy.deepcopy(parent)
        child.fitness = parent.fitness / parent.numerosity
        child.numerosity = 1
        child.experience = 0
        return child

    def _choose_parent(self, classifier_set: ClassifierSet[SymbolType, ActionType]) \
            -> Classifier[SymbolType, ActionType]:
        parent = None

        return parent

    def _mutate(self, classifier: Classifier[SymbolType, ActionType], state: State[SymbolType]):
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
        pass

    def _two_point_crossover(self, cl1: Classifier[SymbolType, ActionType], cl2: Classifier[SymbolType, ActionType]):
        from_index = random.randint(0, len(cl1.condition) - 1)
        to_index = random.randint(0, len(cl1.condition) - 1)

        if from_index > to_index:
            from_index, to_index = to_index, from_index

        self._swap_symbols(cl1.condition, cl2.condition, from_index, to_index)

        return to_index - from_index > 0

    @staticmethod
    def _swap_symbols(condition1: Condition[SymbolType], condition2: Condition[SymbolType],
                      from_index: int, to_index: int):

        for i in range(from_index + 1, to_index):
            condition1[i], condition2[i] = condition2[i], condition1[i]
