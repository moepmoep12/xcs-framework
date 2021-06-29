from typing import Set, List, Iterable, TypeVar

from .classifier import Classifier
from .subsumption import ISubsumptionCriteria
from .exceptions import WrongSubTypeException

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class ClassifierSet(List[Classifier[SymbolType, ActionType]]):
    """
    A ClassifierSet represents a generic collection of classifiers.
    """

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

    def append(self, __object: Classifier[SymbolType, ActionType]) -> None:

        if len(self) < self._max_size:
            super().append(__object)
        else:
            # To-Do: Trigger subsumption or deletion?
            pass

    def extend(self, __iterable: Iterable[Classifier[SymbolType, ActionType]]) -> None:
        super().extend(__iterable)

    def pop(self, __index: int = ...) -> Classifier[SymbolType, ActionType]:
        return super().pop(__index)

    def remove(self, __value: Classifier[SymbolType, ActionType]) -> None:
        super().remove(__value)


class MatchSet(ClassifierSet[SymbolType, ActionType]):
    # TO-DO
    pass


class ActionSet(ClassifierSet[SymbolType, ActionType]):
    # TO-DO
    pass
