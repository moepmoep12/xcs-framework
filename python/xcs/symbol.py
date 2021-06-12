from abc import abstractmethod, ABC
from typing import TypeVar, Generic
from overrides import overrides

SymbolType = TypeVar('SymbolType')
WILDCARD_CHAR = '#'


class ISymbol(Generic[SymbolType], ABC):
    """
    An ISymbol represents the smallest element when representing a generic condition.
    In its simplest form a symbol is just a character.
    """

    @abstractmethod
    def matches(self, value: SymbolType) -> bool:
        """
        :param value: The value to check against.
        :return: Whether this symbol matches to the given value.
        """
        pass


class WildcardSymbol(ISymbol[SymbolType]):
    """
    A WildcardSymbol matches to every other symbol.
    It is represented by the char '#'.
    """

    def matches(self, value: SymbolType) -> bool:
        return value is not None

    def __repr__(self):
        return WILDCARD_CHAR


class Symbol(ISymbol[SymbolType]):
    """
    A generic implementation for a symbol.
    """

    def __init__(self, value: SymbolType):
        self._value: SymbolType = value

    @overrides
    def matches(self, value: SymbolType) -> bool:
        return self.value == value

    @property
    def value(self) -> SymbolType:
        return self._value

    def __repr__(self) -> str:
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.value == other.value
        else:
            return False
