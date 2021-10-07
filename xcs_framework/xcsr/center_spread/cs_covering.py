import copy
import random
from overrides import overrides
from numbers import Number

from xcs_framework.xcs.components.covering import CoveringComponent, SymbolType
from xcs_framework.xcs.symbol import ISymbol
from xcs_framework.xcs.exceptions import WrongSubTypeException

from xcs_framework.xcsr.constants import XCSRCoveringConstants
from xcs_framework.xcsr.center_spread.cs_symbol import CenterSpreadSymbol


class CSCoveringComponent(CoveringComponent):
    """
    Covering component for center spread representation of symbols. Differs from original CoveringComponent by the
    override factory method '_create_symbol' where CSSymbols are instantiated.
    """

    def __init__(self, covering_constants: XCSRCoveringConstants = XCSRCoveringConstants()):
        """
        :param covering_constants: Constants used in this component.
        """
        super(CSCoveringComponent, self).__init__(covering_constants)

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

        center = copy.deepcopy(value)
        spread = random.uniform(0.0, self.covering_constants.max_spread)
        return CenterSpreadSymbol(center=center, spread=spread)
