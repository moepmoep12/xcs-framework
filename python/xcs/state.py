from typing import Tuple, TypeVar

# The data type for symbols
SymbolType = TypeVar('SymbolType')


class State(Tuple[SymbolType]):
    """
    A State is a immutable collection of Symbols.
    """
    pass
