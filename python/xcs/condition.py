from .symbol import ISymbol, WildcardSymbol
from .state import State

from typing import Collection, Generic, TypeVar, Tuple

SymbolType = TypeVar('SymbolType')


class Condition(Generic[SymbolType]):
    """
    A condition consists of an ordered set of symbols.
     A condition can match to a given state.
    """

    def __init__(self, condition: Collection[ISymbol[SymbolType]]):
        if len(condition) == 0:
            raise ValueError("The condition is empty")
        for symbol in condition:
            if not isinstance(symbol, ISymbol):
                raise ValueError(f"Symbol {symbol} is not of type ISymbol")

        self._condition = condition

    def matches(self, state: State[SymbolType]) -> bool:
        """
        Checks this condition against the situation.
        :param state: The state to check against.
        :return: Whether the condition is met in the given situation.
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
        if not isinstance(item, int):
            raise KeyError(f"Key {item} is not of type int")

        if item < 0 or item >= len(self):
            raise KeyError(f"Key {item} is out of range {len(self)}")
        return self.condition[item]

    def __setitem__(self, key: int, value: ISymbol[SymbolType]):
        if not isinstance(key, int):
            raise KeyError(f"Key {key} is not of type int")

        if key < 0 or key >= len(self):
            raise KeyError(f"Key {key} is out of range {len(self)}")

        if not isinstance(value, ISymbol):
            raise ValueError(f"Value {value} is not of type ISymbol")

        self._condition[key] = value
