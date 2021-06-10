from typing import TypeVar, Generic
from overrides import overrides

T = TypeVar('T')
WILDCARD_CHAR = '#'


class ISymbol(Generic[T]):
    """
    An ISymbol represents the smallest element when representing a generic condition.
    In its simplest form a symbol is just a character.
    """

    def matches(self, value: T) -> bool:
        """
        :param value: The value to check against.
        :return: Whether this symbol matches to the given value.
        """
        pass


class WildcardSymbol(ISymbol[T]):
    """
    A WildcardSymbol matches to every other symbol.
    It is represented by the char '#'.
    """

    def matches(self, value: T) -> bool:
        return value is not None

    def __repr__(self):
        return WILDCARD_CHAR


class Symbol(ISymbol[T]):
    """
    A generic implementation for a symbol.
    """

    def __init__(self, value: T):
        self._value: T = value

    @overrides
    def matches(self, value: T) -> bool:
        return self.value == value

    @property
    def value(self) -> T:
        return self._value

    def __repr__(self) -> str:
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.value == other.value
        else:
            return False
