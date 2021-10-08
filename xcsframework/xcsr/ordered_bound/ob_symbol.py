from numbers import Number

from xcsframework.xcs.exceptions import NoneValueException, WrongSubTypeException

from xcsframework.xcsr.bound_symbol import BoundSymbol


class OrderedBoundSymbol(BoundSymbol):
    """
    A bound symbol that is defined by its center and spread.
    """

    def __init__(self, lower: Number, upper: Number):
        """
        :param lower: The lower point of this symbol.
        :param upper: The upper point of this symbol.
        :raises:
            WrongSubtypeException: If lower or upper are not Numbers.
            NoneValueException: If lower or upper is None.
        """

        if lower is None:
            raise NoneValueException(variable_name='lower')
        if upper is None:
            raise NoneValueException(variable_name='upper')

        if not isinstance(upper, Number):
            raise WrongSubTypeException(Number.__name__, type(upper).__name__)
        if not isinstance(lower, Number):
            raise WrongSubTypeException(Number.__name__, type(lower).__name__)

        self._upper = upper
        self._lower = lower

    @property
    def upper_value(self) -> Number:
        return self._upper

    @property
    def lower_value(self) -> Number:
        return self._lower
