from numbers import Number
from math import inf

from xcs_framework.xcs.exceptions import NoneValueException, OutOfRangeException

from xcs_framework.xcsr.bound_symbol import BoundSymbol


class CenterSpreadSymbol(BoundSymbol):
    """
    A bound symbol that is defined by its center and spread.
    """

    def __init__(self, center: Number, spread: Number):
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
    def upper_value(self) -> Number:
        return self._center + self._spread

    @property
    def lower_value(self) -> Number:
        return self._center - self._spread

    @property
    def center(self) -> Number:
        """
        The center point of this symbol.
        :return:
        """
        return self._center

    @property
    def spread(self) -> Number:
        """
        :return: The spread around the center point.
        """
        return self._spread
