from abc import ABCMeta
from typing import Set, TypeVar, Generic, Iterator

from .classifier import Classifier
from .subsumption import ISubsumptionCriteria
from .exceptions import WrongSubTypeException

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

    def insert_classifier(self, __object: Classifier[SymbolType, ActionType], **kwargs) -> None:
        if not isinstance(__object, Classifier):
            raise WrongSubTypeException(Classifier.__name__, type(__object).__name__)

        self._classifier.append(__object)

    def get_available_actions(self) -> Set[ActionType]:
        """
        :return: Returns all unique actions in this collection.
        """
        return set([cl.action for cl in self])


class Population(ClassifierSet[SymbolType, ActionType]):
    """
    A population is a set of classifier that represent the knowledge base of a LCS.
    """

    def __init__(self, max_size: int, subsumption_criteria: ISubsumptionCriteria, *args):
        """
        :param max_size: The maximum size of the population.
        :param subsumption_criteria: Used for determining if a classifier can subsume other classifier.
        :raises:
            AssertionError: If the initial collection is greater than the maximum size.
        """
        assert (len(*args) <= max_size)
        super(Population, self).__init__(*args)
        self._max_size = max_size
        self.subsumption_criteria = subsumption_criteria

    def __len__(self) -> int:
        return sum([cl.numerosity for cl in self])

    def insert_classifier(self, __object: Classifier[SymbolType, ActionType], **kwargs) -> None:
        if not isinstance(__object, Classifier):
            raise WrongSubTypeException(Classifier.__name__, type(__object).__name__)

        do_subsumption = kwargs.get('subsumption', False)

        # population too big -> deletion necessary
        if len(self) + __object.numerosity > self._max_size:
            self._trim_population(desired_size=self._max_size - __object.numerosity)

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

    def _trim_population(self, desired_size: int) -> None:
        pass

    # ------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------------- PROPERTIES ------------------------------------------------ #
    # ------------------------------------------------------------------------------------------------------------- #
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


class MatchSet(ClassifierSet[SymbolType, ActionType]):
    # TO-DO
    pass


class ActionSet(ClassifierSet[SymbolType, ActionType]):
    # TO-DO
    pass
