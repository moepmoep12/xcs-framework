from typing import Set, List, Iterable, TypeVar

from .classifier import Classifier

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
    A population is a set of classifier that represent the knowledge base of the LCS.
    """

    def __init__(self, max_size: int, *args):
        assert (len(*args) <= max_size)
        super(Population, self).__init__(*args)
        self._max_size = max_size

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
    pass


class ActionSet(ClassifierSet[SymbolType, ActionType]):
    pass
