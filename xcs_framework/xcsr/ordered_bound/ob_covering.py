import copy
import random
from overrides import overrides
from numbers import Number

from xcs_framework.xcs.components.covering import CoveringComponent, SymbolType
from xcs_framework.xcs.symbol import ISymbol
from xcs_framework.xcs.exceptions import WrongSubTypeException

from xcs_framework.xcsr.constants import XCSRCoveringConstants
from xcs_framework.xcsr.ordered_bound.ob_symbol import OrderedBoundSymbol


class OBCoveringComponent(CoveringComponent):
    """
    Covering component for center spread representation of symbols. Differs from original CoveringComponent by the
    override factory method '_create_symbol' where CSSymbols are instantiated.
    """

    def __init__(self, covering_constants: XCSRCoveringConstants = XCSRCoveringConstants()):
        """
        :param covering_constants: Constants used in this component.
        """
        super(OBCoveringComponent, self).__init__(covering_constants)

    @overrides
    def _create_symbol(self, value: SymbolType) -> ISymbol[SymbolType]:
        """
        Overrides factory method for symbol creation.
        :param value: The value for the symbol. Can be a ref!
        :return: Newly created symbol.
        :raises:
            WrongSubTypeException: If value is not a Number.
        """
        if not isinstance(value, Number):
            raise WrongSubTypeException(Number.__name__, type(value).__name__)

        lower_min = value - self.covering_constants.max_spread
        upper_max = value + self.covering_constants.max_spread

        # truncate to range
        if self.covering_constants.truncate_to_range:
            lower_min = max(lower_min, self.covering_constants.min_value)
            upper_max = min(upper_max, self.covering_constants.max_value)

        lower_value = random.uniform(lower_min, value)
        upper_value = random.uniform(value, upper_max)

        return OrderedBoundSymbol(lower=lower_value, upper=upper_value)
