from abc import abstractmethod, ABC
from typing import TypeVar, Generic
from overrides import overrides

from .exceptions import NoneValueException

SymbolType = TypeVar('SymbolType')
WILDCARD_CHAR = '#'


class ISymbol(Generic[SymbolType], ABC):
    """
    Interface. An ISymbol represents the smallest element when representing a generic condition.
    In its simplest form a symbol is just a character.
    """

    @abstractmethod
    def matches(self, value: SymbolType) -> bool:
        """
        Checks whether this symbol matches against a given value.

        :param value: The value to check against.
        :return: Whether this symbol matches to the given value.
        """
        pass


class WildcardSymbol(ISymbol[SymbolType]):
    """
    A WildcardSymbol matches to every other value.
    It is represented by the char '#'.
    """

    def matches(self, value: SymbolType) -> bool:
        return value is not None

    def __repr__(self):
        return WILDCARD_CHAR

    def __eq__(self, other):
        return isinstance(other, WildcardSymbol)


class Symbol(ISymbol[SymbolType]):
    """
    A generic implementation for a simple symbol.
    """

    def __init__(self, value: SymbolType):
        """
        :param value: The value of this symbol.
        :raises:
            NoneValueException: If value is None.
        """
        if value is None:
            raise NoneValueException(variable_name='value')
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
        return self.value == getattr(other, 'value', other)
