from typing import TypeVar, Generic
from numbers import Number
from math import inf

from .condition import Condition
from .exceptions import NoneValueException, WrongSubTypeException, OutOfRangeException
from .constants import ClassifierConstants

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


# todo: serialization?
class Classifier(Generic[SymbolType, ActionType]):
    """
    A classifier represents a rule of the form 'if CONDITION then ACTION'.
    """

    def __init__(self,
                 condition: Condition[SymbolType],
                 action: ActionType,
                 cl_constants: ClassifierConstants = ClassifierConstants(),
                 **kwargs
                 ):
        """
        Key worded arguments override values specified in cl_constants.

        :param condition: The condition under which this classifier is active.
        :param action: The action to execute if this classifier is active.
        :param cl_constants: Classifier constants.
        :key f: Custom initial fitness.
        :key exp: Custom initial experience.
        :key n: Custom initial numerosity.
        :key a: Custom initial action set size.
        :key p: Custom initial prediction.
        :key e: Custom initial epsilon.
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
        self._classifier_constants: ClassifierConstants = cl_constants

        self._experience: int = kwargs.get('exp', 0)
        self._numerosity: int = kwargs.get('n', 1)
        self._action_set_size: float = kwargs.get('a', 1)

        self._fitness: float = kwargs.get('f', cl_constants.fitness_init)
        self._prediction: float = kwargs.get('p', cl_constants.prediction_init)
        self._epsilon: float = kwargs.get('e', cl_constants.epsilon_init)

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

    @fitness.setter
    def fitness(self, value: float):
        """
        :param value: The fitness value f of the classifier.
        :raises:
            OutOfRangeException: If value is not in range [0, inf].
        """
        if not isinstance(value, Number) or value < 0:
            raise OutOfRangeException(0, inf, value)

        self._fitness = value

    @property
    def prediction(self) -> float:
        """
        :return: The prediction p for the received reward if the action is executed.
        """
        return self._prediction

    @prediction.setter
    def prediction(self, value: float):
        """
        :param value: The prediction p for the received reward if the action is executed.
        :raises:
            WrongSubTypeException: If value is not a number.
        """
        if not isinstance(value, Number):
            raise WrongSubTypeException(Number.__name__, type(value).__name__)

        self._prediction = value

    @property
    def epsilon(self) -> float:
        """
        :return: The error epsilon of the prediction in range [0, inf].
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        """
        :param value: The error epsilon of the prediction in range [0, inf].
        :raises:
            OutOfRangeException: If value is not in range [0, inf].
        """
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
        """
        Checks whether this classifier could subsume other.

        :param other: Other classifier to check against.
        :return: Whether this classifier subsumes other.
        """
        return self.action == getattr(other, 'action', None) and self.condition.is_more_general(
            getattr(other, 'condition', None))

    def __repr__(self):
        return f"{str(self.condition)} --> {self.action}, " \
               f"F:{self.fitness:.3f}, " \
               f"P:{self.prediction:.3f}, " \
               f"E:{self.epsilon:.3f}, " \
               f"N:{self.numerosity}, " \
               f"exp:{self.experience}, " \
               f"AS:{self.action_set_size:.3f}"
