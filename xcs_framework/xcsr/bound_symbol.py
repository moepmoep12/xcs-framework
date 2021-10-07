from abc import abstractmethod
from numbers import Number
from overrides import overrides

from xcs_framework.xcs.symbol import ISymbol


class BoundSymbol(ISymbol[Number]):
    """
    A BoundSymbol defines a range from lower_value to upper_value in which the symbol does match to a given
    value.
    """

    @overrides
    def matches(self, value: Number) -> bool:
        return self.lower_value <= value <= self.upper_value

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
