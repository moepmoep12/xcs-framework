from .symbol import ISymbol, T

from typing import List, Generic


class Condition(Generic[T]):
    """
    A condition consists of an ordered set of symbols.
     A condition can match to a given situation.
    """

    def __init__(self, condition: List[ISymbol[T]]):
        self._condition = condition

    def matches(self, situation: List[T]) -> bool:
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

    @property
    def condition(self) -> List[ISymbol[T]]:
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
