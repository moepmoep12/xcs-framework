from abc import abstractmethod, ABC
from overrides import overrides
from typing import TypeVar, Collection, Set
import copy
import random

from xcs_framework.xcs.classifier import Classifier
from xcs_framework.xcs.classifier_sets import ClassifierSet
from xcs_framework.xcs.symbol import WildcardSymbol, Symbol
from xcs_framework.xcs.condition import Condition
from xcs_framework.xcs.state import State
from xcs_framework.xcs.selection import IClassifierSelectionStrategy, RouletteWheelSelection
from xcs_framework.xcs.exceptions import *
from xcs_framework.xcs.constants import GAConstants

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class IDiscoveryComponent(ABC):

    @abstractmethod
    def discover(self,
                 timestamp: int,
                 state: State[SymbolType],
                 classifier_set: ClassifierSet[SymbolType, ActionType]) -> ClassifierSet[SymbolType, ActionType]:
        """
        Discovers new classifier based on existing classifier in a specific state.

        :param timestamp: The current timestamp. (for example the current iteration)
        :param state: The current state.
        :param classifier_set: The set used for generating new classifier (== current knowledge base).
        :return: The newly created classifiers.
        """
        pass


# Decorated attribute for classifier.
# Keeps track at which timestamp the GA was called on the Action set the classifier belonged to.
TIMESTAMP = 'timestamp_since_ga'


class GeneticAlgorithm(IDiscoveryComponent):
    """
    A basic Genetic Algorithm (GA) for classifier discovery.
    The GA discovers new classifier from a given set of classifier in a specific state.
    New classifier are created through crossover and mutation.
    """

    def __init__(self,
                 available_actions: Collection[ActionType],
                 selection_strategy: IClassifierSelectionStrategy = RouletteWheelSelection(),
                 ga_constants: GAConstants = GAConstants()):
        """
        :param selection_strategy: The strategy used for selecting parent classifier.
        :param available_actions: Available actions to choose from when mutating the action of a classifier.
        :param ga_constants: Constants used in this ga.
        :raises:
            EmptyCollectionException: If available_actions is empty.
        """

        if available_actions is None or len(available_actions) == 0:
            raise EmptyCollectionException('available_actions')

        self.selection_strategy = selection_strategy
        self._available_actions: Set[ActionType] = set(available_actions)
        self._ga_constants: GAConstants = ga_constants
        # work around for switch-case
        self._crossover_methods = {
            GAConstants.CrossoverMethod.UNIFORM: self._uniform_crossover,
            GAConstants.CrossoverMethod.ONE_POINT: self._one_point_crossover,
            GAConstants.CrossoverMethod.TWO_POINT: self._two_point_crossover
        }

    @overrides
    def discover(self,
                 timestamp: int,
                 state: State[SymbolType],
                 classifier_set: ClassifierSet[SymbolType, ActionType]) -> ClassifierSet[SymbolType, ActionType]:
        """
        Discovers new classifier based on existing classifier in a specific state.

        :param timestamp: The current timestamp. (for example the current iteration)
        :param state: The current state.
        :param classifier_set: The set used for generating new classifier (== current knowledge base).
        :return: The newly created classifiers.
        :raises:
            EmptyCollectionException: If the state or classifier_set is empty.
        """
        if state is None or len(state) == 0:
            raise EmptyCollectionException('state')
        if classifier_set is None or len(classifier_set) == 0:
            raise EmptyCollectionException('classifier_set')

        if not self._should_run(timestamp, classifier_set):
            return ClassifierSet([])

        parent1 = self._choose_parent(classifier_set)
        parent2 = self._choose_parent(classifier_set)

        child1 = self._generate_child(parent1, timestamp)
        child2 = self._generate_child(parent2, timestamp)

        self._crossover(child1, child2)
        self._mutate(child1, state)
        self._mutate(child2, state)

        self._update_timestamps(timestamp, classifier_set)

        return ClassifierSet([child1, child2])

    @property
    def ga_constants(self) -> GAConstants:
        """
        :return: Constants used in this ga.
        """
        return self._ga_constants

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

    def _should_run(self,
                    timestamp: int,
                    classifier_set: ClassifierSet[SymbolType, ActionType]) -> bool:
        """
        :return: Whether the GA should operate on this classifier_set.
                 This is true if the average time since the last GA is greater than a threshold.
        """
        average_timestamp = 0
        numerosity_sum = classifier_set.numerosity_sum()
        for cl in classifier_set:
            # handle classifier that were created outside of this GA
            if not getattr(cl, TIMESTAMP, False):
                # decorating timestamp attribute
                setattr(cl, TIMESTAMP, timestamp)
            average_timestamp += getattr(cl, TIMESTAMP) / numerosity_sum * cl.numerosity

        return timestamp - average_timestamp >= self.ga_constants.ga_threshold

    @staticmethod
    def _update_timestamps(timestamp: int,
                           classifier_set: ClassifierSet[SymbolType, ActionType]) -> None:
        for cl in classifier_set:
            setattr(cl, TIMESTAMP, timestamp)

    @staticmethod
    def _generate_child(parent: Classifier[SymbolType, ActionType], timestamp: int) \
            -> Classifier[SymbolType, ActionType]:
        """
        Generates a child classifier from a parent. Uses deep copy.
        """
        child = Classifier(copy.deepcopy(parent.condition), copy.deepcopy(parent.action))
        child.fitness = parent.fitness / parent.numerosity
        child.prediction = parent.prediction
        child.epsilon = parent.epsilon
        # decorating timestamp attribute
        setattr(child, TIMESTAMP, timestamp)
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
        Turns non-wildcard symbols into wildcards and vice versa.
        """
        for i in range(len(classifier.condition)):
            if random.random() < self.ga_constants.mutation_rate:
                if isinstance(classifier.condition[i], WildcardSymbol):
                    classifier.condition[i] = Symbol(copy.deepcopy(state[i]))
                else:
                    classifier.condition[i] = WildcardSymbol()

        if self.ga_constants.mutate_action:
            actions = list(set(self._available_actions))
            actions.remove(classifier.action)
            if len(actions) > 0:
                classifier._action = actions[random.randint(0, len(actions) - 1)]

    def _crossover(self, cl1: Classifier[SymbolType, ActionType], cl2: Classifier[SymbolType, ActionType]):
        """
        Performs the crossover operation on two classifier.
        """
        did_crossover = False

        if random.random() < self.ga_constants.crossover_probability:
            crossover_method = self._crossover_methods[self.ga_constants.crossover_method]
            did_crossover = crossover_method(cl1, cl2)

        if did_crossover:
            cl1.prediction = cl2.prediction = (cl1.prediction + cl2.prediction) / 2
            cl1.epsilon = cl2.epsilon = (cl1.epsilon + cl2.epsilon) / 2
            cl1.fitness = cl2.fitness = (cl1.fitness + cl2.fitness) / 2

        cl1.fitness *= self.ga_constants.fitness_reduction
        cl2.fitness *= self.ga_constants.fitness_reduction

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
