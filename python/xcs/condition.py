from .symbol import ISymbol, SymbolType, WildcardSymbol

from typing import List, Generic


class Condition(Generic[SymbolType]):
    """
    A condition consists of an ordered set of symbols.
     A condition can match to a given situation.
    """

    def __init__(self, condition: List[ISymbol[SymbolType]]):
        self._condition = condition

    def matches(self, situation: List[SymbolType]) -> bool:
        """
        Checks this condition against the situation.
        :param situation: The situation to check against.
        :return: Whether the condition is met in the given situation.
        """
        assert (len(situation) == len(self._condition))

        for i in range(len(self._condition)):
            if not self._condition[i].matches(situation[i]):
                return False

        return True

    def is_more_general(self, other) -> bool:
        """
        Checks whether this condition is more general than another condition.
        :param other: The condition to check against.
        :raises: AssertionError if other has not the same length.
        :return: Whether this condition is more general.
        """
        assert (len(self.condition) == len(other.condition))

        result = False

        for i in range(len(self.condition)):
            if self.condition[i] != other.condition[i]:
                if not isinstance(self.condition[i], WildcardSymbol):
                    return False
                else:
                    result = True

        return result

    @property
    def condition(self) -> List[ISymbol[SymbolType]]:
        return self._condition

    def __repr__(self):
        result: str = '['
        for i, symbol in enumerate(self._condition):
            result += f"{str(symbol)}{'|' if i < len(self._condition) - 1 else ''}"
        return f"{result}]"

    def __len__(self) -> int:
        return len(self._condition)

    def __eq__(self, o: object) -> bool:
        return self.condition == getattr(o, 'condition', None)

    def __getitem__(self, item):
        return self.condition[item]
