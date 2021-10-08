from .symbol import ISymbol, WildcardSymbol
from .state import State
from .exceptions import EmptyCollectionException, WrongSubTypeException, OutOfRangeException, WrongStrictTypeException

from typing import Collection, Generic, TypeVar, Tuple

SymbolType = TypeVar('SymbolType')


class Condition(Generic[SymbolType]):
    """
    A condition consists of an ordered set of symbols.
    A condition can match to a given state.
    """

    def __init__(self, condition: Collection[ISymbol[SymbolType]]):
        """
        :param condition: A collection of ISymbol representing the value of this condition.
        :raises:
            EmptyCollectionException: If condition is empty.
            WrongSubTypeException: If one element of condition is not a ISymbol.
        """
        if condition is None or len(condition) == 0:
            raise EmptyCollectionException(variable_name='condition')

        # check if symbols are valid
        for symbol in condition:
            if not isinstance(symbol, ISymbol):
                raise WrongSubTypeException(expected=ISymbol.__name__, actual=type(symbol).__name__)

        self._condition = condition

    def matches(self, state: State[SymbolType]) -> bool:
        """
        Checks whether this condition matches to the given state.

        :param state: The state to check against.
        :return: Whether the condition is met in the given situation.
        :raises:
            AssertionError: If the lengths are not equal.
        """
        assert (len(state) == len(self._condition))

        for i in range(len(self._condition)):
            if not self._condition[i].matches(state[i]):
                return False

        return True

    def is_more_general(self, other) -> bool:
        """
        Checks whether this condition is more general than another condition.

        :param other: The condition to check against.
        :raises: AssertionError if other has not the same length.
        :return: Whether this condition is more general.
        :raises:
            AssertionError: If the lengths are not equal.
        """
        assert (len(self.condition) == len(other.condition))

        result = False

        for i in range(len(self.condition)):
            if self.condition[i] != other.condition[i]:
                if not isinstance(self.condition[i], WildcardSymbol):
                    return False
                else:
                    result = True

        return result

    @property
    def condition(self) -> Tuple[ISymbol[SymbolType]]:
        """
        :return: The value of this condition (immutable).
        """
        return tuple(self._condition)

    def __repr__(self):
        result: str = '['
        for i, symbol in enumerate(self._condition):
            result += f"{str(symbol)}{'|' if i < len(self._condition) - 1 else ''}"
        return f"{result}]"

    def __len__(self) -> int:
        return len(self._condition)

    def __eq__(self, o: object) -> bool:
        return self.condition == getattr(o, 'condition', None)

    def __getitem__(self, item: int):
        """
        :param item: The index.
        :return: The element at the index 'item'.
        :raises:
            WrongStrictTypeException: If item is not an int.
            OutOfRangeException: If item is not in range [0, len(self) - 1].
        """
        if not isinstance(item, int):
            raise WrongStrictTypeException(expected=int.__name__, actual=type(item).__name__)
        if item < 0 or item >= len(self):
            raise OutOfRangeException(0, len(self) - 1, item)

        return self.condition[item]

    def __setitem__(self, key: int, value: ISymbol[SymbolType]):
        """
        :param key: The index.
        :param value: The value to set.
        :raises:
            WrongStrictTypeException: If key is not an int.
            OutOfRangeException: If item is not in range [0, len(self) - 1].
            WrongSubTypeException: If value is not of type ISymbol.
        """
        if not isinstance(key, int):
            raise WrongStrictTypeException(expected=int.__name__, actual=type(key).__name__)

        if key < 0 or key >= len(self):
            raise OutOfRangeException(0, len(self) - 1, key)

        if not isinstance(value, ISymbol):
            raise WrongSubTypeException(expected=ISymbol.__name__, actual=type(value).__name__)

        self._condition[key] = value
