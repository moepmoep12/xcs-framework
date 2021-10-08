from abc import abstractmethod, ABC
from typing import TypeVar, List
from numbers import Number

from xcsframework.xcs.state import State

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class IEnvironment(ABC):
    """
    Interface for an Environment.
    """

    @abstractmethod
    def get_state(self) -> State[SymbolType]:
        """
        :return: The current state.
        """
        pass

    @abstractmethod
    def get_available_actions(self) -> List[ActionType]:
        """
        Actions that can be executed in this environment.
        """
        pass

    @abstractmethod
    def execute_action(self, action: ActionType) -> Number:
        """
        :param action: The action to execute.
        :return: The reward received for executing the action.
        """
        pass

    @abstractmethod
    def is_end_of_problem(self) -> bool:
        """
        :return: Whether the problem was solved by the previous executed action.
        For single step problems this is always true!
        """
        pass
