from abc import abstractmethod, ABC
from overrides import overrides
from typing import TypeVar, List, Collection, Set
import copy
import random
from numbers import Number

from xcs.classifier import Classifier
from xcs.classifier_sets import ClassifierSet
from xcs.symbol import WildcardSymbol
from xcs.condition import Condition
from xcs.state import State
from xcs.selection import IClassifierSelectionStrategy
from xcs.exceptions import *

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class IDiscoveryComponent(ABC):

    @abstractmethod
    def discover(self,
                 state: State[SymbolType],
                 classifier_set: ClassifierSet[SymbolType, ActionType]) -> ClassifierSet[SymbolType, ActionType]:
        """
        Discovers new classifier based on existing classifier in a specific state.

        :param state: The current state.
        :param classifier_set: The set used for generating new classifier (== current knowledge base).
        :return: The newly created classifiers.
        """
        pass


class GeneticAlgorithm(IDiscoveryComponent):
    """
    A basic Genetic Algorithm (GA) for classifier discovery.
    The GA discovers new classifier from a given set of classifier in a specific state.
    New classifier are created through crossover and mutation.
    """

    # TO-DO: Add constructor arguments
    def __init__(self,
                 selection_strategy: IClassifierSelectionStrategy,
                 available_actions: Collection[ActionType],
                 mutation_rate: float = 0.03,
                 mutate_action: bool = False,
                 fitness_reduction: float = 0.1,
                 crossover_probability: float = 0.5):
        """
        :param selection_strategy: The strategy used for selecting parent classifier.
        :param available_actions: Available actions to choose from when mutating the action of a classifier.
        :param mutation_rate: The value of the rate of mutation as a float in range [0.0, 1.0].
        :param mutate_action: Whether the action of a classifier has a chance to be mutated during discovery.
        :param fitness_reduction: Float in range [0.0, 1.0] indicating how much the fitness of a child classifier
                                  will be reduced when it is created without crossover.
        :param crossover_probability: The chance in the range [0.0, 1.0] for doing crossover in classifier discovery.
        :raises:
            EmptyCollectionException: If available_actions is empty.
        """

        if available_actions is None or len(available_actions) == 0:
            raise EmptyCollectionException('available_actions')

        self.mutation_rate = mutation_rate
        self.mutate_action = mutate_action
        self.selection_strategy = selection_strategy
        self.fitness_reduction = fitness_reduction
        self.crossover_probability = crossover_probability
        self._available_actions: Set[ActionType] = set(available_actions)

    @overrides
    def discover(self,
                 state: State[SymbolType],
                 classifier_set: ClassifierSet[SymbolType, ActionType]) -> ClassifierSet[SymbolType, ActionType]:

        if state is None or len(state) == 0:
            raise EmptyCollectionException('state')
        if classifier_set is None or len(classifier_set) == 0:
            raise EmptyCollectionException('classifier_set')

        parent1 = self._choose_parent(classifier_set)
        parent2 = self._choose_parent(classifier_set)

        child1 = self._generate_child(parent1)
        child2 = self._generate_child(parent2)

        self._crossover(child1, child2)
        self._mutate(child1, state)
        self._mutate(child2, state)

        return ClassifierSet([child1, child2])

    @property
    def mutation_rate(self) -> float:
        """
        :return: The rate of mutation when discovering new classifier. In range [0.0, 1.0].
        """
        return self._mutation_rate

    @mutation_rate.setter
    def mutation_rate(self, value: float):
        """
        :param value: The value of the rate of mutation as a float in range [0.0, 1.0].
        :raises:
            OutOfRangeException: If value is not a float in range [0.0, 1.0]
        """
        if not isinstance(value, Number) or value < 0.0 or value > 1.0:
            raise OutOfRangeException(0.0, 1.0, value)

        self._mutation_rate = value

    @property
    def mutate_action(self) -> bool:
        """
        :return: Whether the action will be mutated when discovering new classifier.
        """
        return self._mutate_action

    @mutate_action.setter
    def mutate_action(self, value: bool):
        """
        :param value: Whether the action of a classifier has a chance to be mutated during discovery.
        """
        self._mutate_action = value

    @property
    def selection_strategy(self) -> IClassifierSelectionStrategy:
        """
        :return: The strategy used for selecting parent classifier.
        """
        return self._selection_strategy

    @selection_strategy.setter
    def selection_strategy(self, value: IClassifierSelectionStrategy):
        """
        :param value: The strategy used for selecting parent classifier.
        :raises:
            WrongSubTypeException: If value is not of type IClassifierSelectionStrategy.
        """
        if not isinstance(value, IClassifierSelectionStrategy):
            raise WrongSubTypeException(IClassifierSelectionStrategy.__name__, type(value).__name__)

        self._selection_strategy = value

    @property
    def fitness_reduction(self) -> float:
        """
        :return: The percentage reduction of the fitness of a child classifier when created without crossover.
                 In range [0.0, 1.0].
        """
        return self._fitness_reduction

    @fitness_reduction.setter
    def fitness_reduction(self, value: float):
        """
        :param value: Float in range [0.0, 1.0].
        : raises:
            OutOfRangeException: If value is not a float in range [0.0, 1.0].
        """
        if not isinstance(value, Number) or value < 0.0 or value > 1.0:
            raise OutOfRangeException(0.0, 1.0, value)

        self._fitness_reduction = value

    @property
    def crossover_probability(self) -> float:
        """
        :return: The chance in the range [0.0, 1.0] for doing crossover in classifier discovery.
        """
        return self._crossover_probability

    @crossover_probability.setter
    def crossover_probability(self, value: float):
        """
        :param value: Float in range [0.0, 1.0].
        : raises:
            OutOfRangeException: If value is not a float in range [0.0, 1.0].
        """
        if not isinstance(value, Number) or value < 0.0 or value > 1.0:
            raise OutOfRangeException(0.0, 1.0, value)

        self._crossover_probability = value

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
        return classifier_set[self.selection_strategy.select_classifier(classifier_set, lambda cl: cl.fitness)]

    def _mutate(self, classifier: Classifier[SymbolType, ActionType], state: State[SymbolType]):
        """
        Mutates a classifier by changing some of its condition symbols and the action if enabled.
        """
        for i in range(len(classifier.condition)):
            if random.random() < self.mutation_rate:
                if isinstance(classifier.condition[i], WildcardSymbol):
                    classifier.condition[i] = state[i]
                else:
                    classifier.condition[i] = WildcardSymbol()

        if self.mutate_action:
            actions = set(self.available_actions)
            actions.remove(classifier.action)
            if len(actions) > 0:
                classifier.action = actions[random.randint(0, len(actions) - 1)]

    def _crossover(self, cl1: Classifier[SymbolType, ActionType], cl2: Classifier[SymbolType, ActionType]):
        """
        Performs the crossover operation on two classifier.
        """
        performed_crossover = False

        if random.random() < self.crossover_probability:
            # TO-DO: Perform crossover according to strategy
            pass

        if performed_crossover:
            cl1.prediction = cl2.prediction = (cl1.prediction + cl2.prediction) / 2
            cl1.epsilon = cl2.epsilon = (cl1.epsilon + cl2.epsilon) / 2
            cl1.fitness = cl2.fitness = (cl1.fitness + cl2.fitness) / 2

        cl1.fitness *= self.fitness_reduction
        cl2.fitness *= self.fitness_reduction

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
            condition1[i], condition2[i] = condition2[i], condition1[i]
            swapped = True

        return swapped
