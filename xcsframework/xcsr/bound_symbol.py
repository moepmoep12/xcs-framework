from abc import abstractmethod
from numbers import Number
from overrides import overrides

from xcsframework.xcs.symbol import ISymbol, ComparisonResult, WildcardSymbol
from xcsframework.xcs.exceptions import NoneValueException


class BoundSymbol(ISymbol[Number]):
    """
    A BoundSymbol defines a range from lower_value to upper_value in which the symbol does match to a given
    value.
    """

    @overrides
    def matches(self, value: Number) -> bool:
        return self.lower_value <= value <= self.upper_value

    @overrides
    def compare(self, other) -> ComparisonResult:
        if isinstance(other, WildcardSymbol):
            return ComparisonResult.LESS_GENERAL

        lower = getattr(other, 'lower_value', None)
        upper = getattr(other, 'upper_value', None)
        if lower is None:
            raise NoneValueException('other.lower_value')
        if upper is None:
            raise NoneValueException('other.upper_value')

        if lower == self.lower_value and upper == self.upper_value:
            return ComparisonResult.EQUAL

        if self.lower_value <= lower and self.upper_value >= upper:
            # we are enclosing other
            return ComparisonResult.MORE_GENERAL
        elif self.lower_value >= lower and self.upper_value <= upper:
            # we are enclosed by other
            return ComparisonResult.LESS_GENERAL

        return ComparisonResult.UNDECIDABLE

    @abstractmethod
    def upper_value(self) -> Number:
        pass

    @abstractmethod
    def lower_value(self) -> Number:
        pass

    def __repr__(self) -> str:
        return f"({self.lower_value:.2f} - {self.upper_value:.2f})"

    def __eq__(self, other):
        return self.upper_value == getattr(other, 'upper_value', other) \
               and self.lower_value == getattr(other, 'lower_value', other)
