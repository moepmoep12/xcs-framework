from typing import TypeVar, Generic
from numbers import Number
from math import inf

from .condition import Condition
from .exceptions import NoneValueException, WrongSubTypeException, OutOfRangeException

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class Classifier(Generic[SymbolType, ActionType]):
    """
    A classifier represents a rule of the form 'if CONDITION then ACTION'.
    """

    def __init__(self, condition: Condition[SymbolType], action: ActionType):
        """
        :param condition: The condition under which this classifier is active.
        :param action: The action to execute if this classifier is active.
        :raises:
            NoneValueException: If any of the required arguments is None.
            WrongSubTypeException: If the condition is not of type Condition.
        """
        if condition is None:
            raise NoneValueException('condition')
        if action is None:
            raise NoneValueException('action')
        if not isinstance(condition, Condition):
            raise WrongSubTypeException(Condition.__name__, type(condition).__name__)

        self._condition: Condition[SymbolType] = condition
        self._action: ActionType = action
        self._experience: int = 0
        self._fitness: float = 0
        self._numerosity: int = 1
        self._prediction: float = 0
        self._epsilon: float = 0
        self._action_set_size: float = 1

    @property
    def condition(self) -> Condition[SymbolType]:
        """
        :return: The condition under which this classifier is active.
        """
        return self._condition

    @property
    def action(self) -> ActionType:
        """
        :return: The suggested action (or classification) to take if the condition is met.
        """
        return self._action

    @property
    def fitness(self) -> float:
        """
        :return: The fitness value f of the classifier.
        """
        return self._fitness

    # To-Do: Restrict to positive values only?
    @fitness.setter
    def fitness(self, value: float):
        self._fitness = value

    @property
    def prediction(self) -> float:
        """
        :return: The prediction for the received reward if the action is executed.
        """
        return self._prediction

    @prediction.setter
    def prediction(self, value: float):
        self._prediction = value

    @property
    def epsilon(self) -> float:
        """
        :return: The error epsilon of the prediction in range [0, inf].
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        if not isinstance(value, Number) or value < 0:
            raise OutOfRangeException(0.0, inf, value)
        self._epsilon = value

    @property
    def numerosity(self) -> int:
        """
        :return: How many classifier this classifier represents.
        Increases by the process of subsumption.
        """
        return self._numerosity

    @numerosity.setter
    def numerosity(self, value: int):
        """
        :param value: How many classifier this classifier represents.
        :raises:
            OutOfRangeException: If value is not in range [1, inf].
        """
        if not isinstance(value, Number) or value < 1:
            raise OutOfRangeException(1, inf, value)
        self._numerosity = value

    @property
    def action_set_size(self) -> float:
        """
        :return: The average size of the action set this classifier belonged to.
        """
        return self._action_set_size

    @action_set_size.setter
    def action_set_size(self, value: float):
        """
        :param value: The average size of the action set this classifier belonged to.
        :raises:
            OutOfRangeException: If value is not in range [1, inf].
        """
        if not isinstance(value, Number) or value < 1:
            raise OutOfRangeException(1, inf, value)

        self._action_set_size = value

    @property
    def experience(self) -> int:
        """
        :return: How often this classifier belonged to the action set.
        """
        return self._experience

    def increment_experience(self):
        self._experience += 1

    def subsumes(self, other) -> bool:
        # TO-DO: Exceptions?
        """
        Checks whether this classifier could subsume other.

        :param other: Other classifier to check against.
        :return: Whether this classifier subsumes other.
        """
        return self.action == getattr(other, 'action', None) \
               and self.condition.is_more_general(getattr(other, 'condition', None))

    def __repr__(self):
        return f"{str(self.condition)} : {self.action}, F:{self.fitness}, P:{self.prediction}, E:{self.epsilon}," \
               f" N:{self.numerosity}, exp:{self.experience} "
