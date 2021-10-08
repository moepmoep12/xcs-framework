from math import inf
from numbers import Number

from xcs_framework.xcs.constants import CoveringConstants, GAConstants
from xcs_framework.xcs.exceptions import OutOfRangeException, WrongStrictTypeException


class XCSRCoveringConstants(CoveringConstants):
    """
    Extension of CoveringConstants for real valued symbol representation.
    """

    def __init__(self, max_spread: Number = 1.0, min_value: Number = 0.0, max_value: Number = 1.0,
                 truncate_to_range: bool = False):
        """
        :param max_spread: The maximum spread around the center when creating a new symbol. Number >= 0.
        :param min_value: Minimum allowed value for the lower_value of a bound symbol.
        :param max_value: Maximum allowed value for the upper_value of a bound symbol.
        :param truncate_to_range: Whether the bounds of a symbol will be truncated into the range [min_value, max_value]
        """
        super(XCSRCoveringConstants, self).__init__(wild_card_probability=0)
        self.max_spread = max_spread
        self.min_value = min_value
        self.max_value = max_value
        self.truncate_to_range = truncate_to_range

    @property
    def max_spread(self) -> Number:
        """
        :return: Maximum spread around a value when creating a new symbol.
        """
        return self._max_spread

    @max_spread.setter
    def max_spread(self, value: Number):
        """
        :param value: Maximum spread around a value when creating a new symbol. Number in range [0.0, inf].
        :raises:
            OutOfRangeException: If value is not a number in range [0.0, inf].
        """
        if not isinstance(value, Number) or value < 0.0:
            raise OutOfRangeException(0.0, inf, value)
        self._max_spread = value

    @property
    def min_value(self) -> Number:
        """
        :return: Minimum allowed value for the lower_value of a bound symbol.
        """
        return self._min_value

    @min_value.setter
    def min_value(self, value: Number):
        """
        :param value: Maximum allowed value for the upper_value of a bound symbol.
        :raises:
            OutOfRangeException: If value is not a number in range [-inf, inf].
        """
        if not isinstance(value, Number):
            raise OutOfRangeException(-inf, inf, value)
        self._min_value = value

    @property
    def max_value(self) -> Number:
        """
        :return: Minimum allowed value for the lower_value of a bound symbol.
        """
        return self._max_value

    @max_value.setter
    def max_value(self, value: Number):
        """
        :param value: Maximum allowed value for the upper_value of a bound symbol.
        :raises:
            OutOfRangeException: If value is not a number in range [-inf, inf].
        """
        if not isinstance(value, Number):
            raise OutOfRangeException(-inf, inf, value)
        self._max_value = value

    @property
    def truncate_to_range(self) -> bool:
        """
        :return: Whether the bounds of a symbol will be truncated into the range [min_value, max_value]
        """
        return self._truncate_to_range

    @truncate_to_range.setter
    def truncate_to_range(self, value: bool):
        """
        :param value: Whether the bounds of a symbol will be truncated into the range [min_value, max_value]
        :raises:
            WrongStrictTypeException: If value is not of type bool.
        """
        if not isinstance(value, bool):
            raise WrongStrictTypeException(bool.__name__, type(value).__name__)
        self._truncate_to_range = value


class XCSRGAConstants(GAConstants):
    """
    Extension of GAConstants for real valued symbol representation.
    """

    def __init__(self,
                 ga_constants: GAConstants = GAConstants(),
                 max_mutation_change: Number = 0.1,
                 min_value: Number = 0.0,
                 max_value: Number = 1.0,
                 truncate_to_range: bool = False):
        """

        :param ga_constants: GA constants.
        :param max_mutation_change: The maximum change of center/spread of a symbol in the mutation operation.
         Number >= 0.
        :param min_value: Minimum allowed value for the lower_value of a bound symbol.
        :param max_value: Maximum allowed value for the upper_value of a bound symbol.
        :param truncate_to_range: Whether the bounds of a symbol will be truncated into the range [min_value, max_value]
        """
        super(XCSRGAConstants, self).__init__(mutation_rate=ga_constants.mutation_rate,
                                              mutate_action=ga_constants.mutate_action,
                                              fitness_reduction=ga_constants.fitness_reduction,
                                              crossover_probability=ga_constants.crossover_probability,
                                              ga_threshold=ga_constants.ga_threshold,
                                              crossover_method=ga_constants.crossover_method)
        self.max_mutation_change = max_mutation_change
        self.min_value = min_value
        self.max_value = max_value
        self.truncate_to_range = truncate_to_range

    @property
    def max_mutation_change(self) -> Number:
        """
        :return: The maximum change of center/spread of a symbol in the mutation operation.
        """
        return self._max_mutation_change

    @max_mutation_change.setter
    def max_mutation_change(self, value: Number):
        """
        :param value: The maximum change of center/spread of a symbol in the mutation operation.
         Number in range [0.0, inf].
        :raises:
            OutOfRangeException: If value is not a number in range [0.0, inf].
        """
        if not isinstance(value, Number) or value < 0.0:
            raise OutOfRangeException(0.0, inf, value)
        self._max_mutation_change = value

    @property
    def min_value(self) -> Number:
        """
        :return: Minimum allowed value for the lower_value of a bound symbol.
        """
        return self._min_value

    @min_value.setter
    def min_value(self, value: Number):
        """
        :param value: Maximum allowed value for the upper_value of a bound symbol.
        :raises:
            OutOfRangeException: If value is not a number in range [-inf, inf].
        """
        if not isinstance(value, Number):
            raise OutOfRangeException(-inf, inf, value)
        self._min_value = value

    @property
    def max_value(self) -> Number:
        """
        :return: Minimum allowed value for the lower_value of a bound symbol.
        """
        return self._max_value

    @max_value.setter
    def max_value(self, value: Number):
        """
        :param value: Maximum allowed value for the upper_value of a bound symbol.
        :raises:
            OutOfRangeException: If value is not a number in range [-inf, inf].
        """
        if not isinstance(value, Number):
            raise OutOfRangeException(-inf, inf, value)
        self._max_value = value

    @property
    def truncate_to_range(self) -> bool:
        """
        :return: Whether the bounds of a symbol will be truncated into the range [min_value, max_value]
        """
        return self._truncate_to_range

    @truncate_to_range.setter
    def truncate_to_range(self, value: bool):
        """
        :param value: Whether the bounds of a symbol will be truncated into the range [min_value, max_value]
        :raises:
            WrongStrictTypeException: If value is not of type bool.
        """
        if not isinstance(value, bool):
            raise WrongStrictTypeException(bool.__name__, type(value).__name__)
        self._truncate_to_range = value
