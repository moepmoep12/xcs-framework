from abc import abstractmethod, ABC
from typing import TypeVar, Generic
from overrides import overrides
from math import inf
from numbers import Number

from .exceptions import NoneValueException, OutOfRangeException
from .constants import SymbolConstants

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


class BoundSymbol(ISymbol[SymbolType]):
    """
    A BoundSymbol defines a range from lower_value to upper_value in which the symbol does match to a given
    value.
    """

    @overrides
    def matches(self, value: SymbolType) -> bool:
        return self.lower_value <= value <= self.upper_value

    @abstractmethod
    def upper_value(self) -> SymbolType:
        pass

    @abstractmethod
    def lower_value(self) -> SymbolType:
        pass

    def __repr__(self) -> str:
        return f"({self.lower_value} - {self.upper_value})"

    def __eq__(self, other):
        return self.upper_value == getattr(other, 'upper_value', other) \
               and self.lower_value == getattr(other, 'lower_value', other)


class CenterSpreadSymbol(BoundSymbol[SymbolType]):
    """
    A bound symbol that is defined by its center and spread.
    """

    def __init__(self, center: SymbolType, spread: SymbolType):
        """
        :param center: The center point of this symbol.
        :param spread: The spread around the center. Must be >= 0.
        :raises:
            OutOfRangeException: If spread is < 0 .
            NoneValueException: If center or spread is None.
        """

        if center is None:
            raise NoneValueException(variable_name='center')
        if spread is None:
            raise NoneValueException(variable_name='spread')

        if not isinstance(spread, Number) or spread < 0.0:
            raise OutOfRangeException(0.0, inf, spread)

        self._center = center
        self._spread = spread

    @property
    def upper_value(self) -> SymbolType:
        return self._center + self._spread

    @property
    def lower_value(self) -> SymbolType:
        return self._center - self._spread


# dict used for getting the constructor of a symbol representation
SYMBOL_CONSTRUCTORS = {
    SymbolConstants.SymbolRepresentation.NORMAL: Symbol,
    SymbolConstants.SymbolRepresentation.CENTER_SPREAD: CenterSpreadSymbol
}
