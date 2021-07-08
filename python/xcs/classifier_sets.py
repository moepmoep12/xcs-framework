from typing import Set, TypeVar, Generic, Iterator
from numbers import Number
from math import inf

from .classifier import Classifier
from .subsumption import ISubsumptionCriteria
from .exceptions import WrongSubTypeException, OutOfRangeException
from .selection import IClassifierSelectionStrategy, RouletteWheelSelection

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class ClassifierSet(Generic[SymbolType, ActionType]):
    """
    A ClassifierSet represents a generic collection of classifiers.
    """

    def __init__(self, *args):
        self._classifier = list(*args)

    def __len__(self) -> int:
        return len(self._classifier)

    def __iter__(self) -> Iterator[Classifier[SymbolType, ActionType]]:
        return self._classifier.__iter__()

    def __contains__(self, __x: object) -> bool:
        return self._classifier.__contains__(__x)

    def __getitem__(self, item):
        return self._classifier.__getitem__(item)

    def __eq__(self, other):
        return self._classifier.__eq__(getattr(other, '_classifier', None))

    def __repr__(self) -> str:
        return self._classifier.__repr__()

    def insert_classifier(self, __object: Classifier[SymbolType, ActionType], **kwargs) -> None:
        """
        Inserts a classifier into this set.
        :param __object: The classifier to be inserted.
        :param kwargs: Key worded arguments.
        :raises:
            WrongSubTypeException: If __object is not a subtype of Classifier.
        """
        if not isinstance(__object, Classifier):
            raise WrongSubTypeException(Classifier.__name__, type(__object).__name__)

        self._classifier.append(__object)

    def remove_classifier(self, classifier: Classifier[SymbolType, ActionType]) -> None:
        """
        Removes the classifier from this set.

        :param classifier: The classifier to remove.
        :raises:
            ValueError: If the classifier is not present in this set.
        """
        self._classifier.remove(classifier)

    def get_available_actions(self) -> Set[ActionType]:
        """
        :return: Returns all unique actions in this collection.
        """
        return set([cl.action for cl in self])

    def numerosity_sum(self) -> int:
        """
        :return: The sum of the numerosity of all classifier.
        """
        return sum(cl.numerosity for cl in self)


class Population(ClassifierSet[SymbolType, ActionType]):
    """
    A population is a set of classifier that represent the knowledge base of a LCS.
    """

    # todo: use settings/params dataclass?
    def __init__(self, max_size: int,
                 subsumption_criteria: ISubsumptionCriteria,
                 deletion_selection: IClassifierSelectionStrategy = RouletteWheelSelection(),
                 classifier=[]):
        """
        :param max_size: The maximum size of the population.
        :param subsumption_criteria: Used for determining if a classifier can subsume other classifier.
        :param deletion_selection: The strategy used for selection classifier to delete from population.
        :raises:
            AssertionError: If the initial collection is greater than the maximum size.
        """
        assert (len(classifier) <= max_size)
        super(Population, self).__init__(classifier)
        self.deletion_selection = deletion_selection
        self._max_size = max_size
        self.subsumption_criteria = subsumption_criteria

        self._deletion_exp_threshold = 20
        self._deletion_fitness_fraction = 0.1

    # todo: docstring?
    def insert_classifier(self, __object: Classifier[SymbolType, ActionType], **kwargs) -> None:
        if not isinstance(__object, Classifier):
            raise WrongSubTypeException(Classifier.__name__, type(__object).__name__)

        do_subsumption = kwargs.get('subsumption', False)

        # population too big -> deletion necessary
        if self.numerosity_sum() + __object.numerosity > self._max_size:
            self.trim_population(desired_size=self._max_size - __object.numerosity)

        # check for same classifier
        for cl in self:
            if cl.condition == __object.condition and cl.action == __object.action:
                cl.numerosity += __object.numerosity
                return

        # classifier isn't present, check for subsumption
        if do_subsumption:
            for cl in self:
                if cl.subsumes(__object) and self._subsumption_criteria.can_subsume(cl):
                    cl.numerosity += __object.numerosity
                    return

        # classifier is new, add it
        self._classifier.append(__object)

    # todo: docstring
    def trim_population(self, desired_size: int) -> None:

        while numerosity_sum := self.numerosity_sum() > desired_size:
            average_fitness = sum([cl.fitness for cl in self]) / numerosity_sum

            def deletion_vote(cl: Classifier[SymbolType, ActionType]) -> float:
                vote = cl.action_set_size * cl.numerosity

                if average_fitness > 0 and cl.fitness > 0 and cl.experience > self._deletion_exp_threshold and (
                        cl.fitness / cl.numerosity) < self._deletion_fitness_fraction * average_fitness:
                    vote *= average_fitness / (cl.fitness / cl.numerosity)
                return vote

            index = self.deletion_selection.select_classifier(self._classifier, deletion_vote)

            classifier = self[index]

            if classifier.numerosity > 1:
                classifier.numerosity -= 1
            else:
                del self._classifier[index]

    @property
    def max_size(self) -> int:
        """
        :return: The maximum size of this population. The size is measured in total numerosity.
        """
        return self._max_size

    @max_size.setter
    def max_size(self, value: int):
        """
        If value < current max_size then deletion will occur.
        :param value: The maximum size of the population in range [1, inf].
        """
        if not isinstance(value, Number) or value < 1:
            raise OutOfRangeException(1, inf, value)

        self._max_size = value
        self.trim_population(value)

    @property
    def subsumption_criteria(self) -> ISubsumptionCriteria:
        """
        :return: The object used for determining if a classifier can do subsumption.
        """
        return self._subsumption_criteria

    @subsumption_criteria.setter
    def subsumption_criteria(self, value: ISubsumptionCriteria):
        """
        :param value: Object that implements ISubsumptionCriteria.
        :raises:
            WrongSubTypeException: If value is not a subtype of ISubsumptionCriteria.
        """
        if not isinstance(value, ISubsumptionCriteria):
            raise WrongSubTypeException(ISubsumptionCriteria.__name__, type(value).__name__)

        self._subsumption_criteria = value

    @property
    def deletion_selection(self) -> IClassifierSelectionStrategy:
        """
        :return: The strategy used for selection classifier to delete from population.
        """
        return self._deletion_selection

    @deletion_selection.setter
    def deletion_selection(self, value: IClassifierSelectionStrategy):
        """
        :param value: Object that implements IClassifierSelectionStrategy.
        :raises:
            WrongSubTypeException: If value is not a subtype of IClassifierSelectionStrategy.
        """
        if not isinstance(value, IClassifierSelectionStrategy):
            raise WrongSubTypeException(IClassifierSelectionStrategy.__name__, type(value).__name__)

        self._deletion_selection = value


class MatchSet(ClassifierSet[SymbolType, ActionType]):
    # TODO
    pass


class ActionSet(ClassifierSet[SymbolType, ActionType]):
    # TODO
    pass
