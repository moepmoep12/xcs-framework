from typing import TypeVar, Generic

from .condition import Condition

# The data type for symbols
SymbolType = TypeVar('SymbolType')
# The data type for actions
ActionType = TypeVar('ActionType')


class Classifier(Generic[SymbolType, ActionType]):
    """
    A classifier represents a rule of the form 'if CONDITION then ACTION'.
    """

    def __init__(self, condition: Condition[SymbolType], action: ActionType):
        if condition is None:
            raise ValueError("Condition argument is None")
        if not isinstance(condition, Condition):
            raise ValueError(f"Condition argument is not of type Condition but {type(condition)}")
        if action is None:
            raise ValueError("Action argument is None")

        self._condition: Condition[SymbolType] = condition
        self._action: ActionType = action
        self._experience: int = 0
        self._fitness: float = 0
        self._numerosity: int = 1
        self._prediction: float = 0
        self._epsilon: float = 0

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
        :return: The error epsilon of the prediction.
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        self._epsilon = value

    @property
    def numerosity(self) -> int:
        """
        :return: How many classifier this classifier represents.
        Increases by the process of subsumption.
        """
        return self._numerosity

    @property
    def experience(self) -> int:
        """
        :return: How often this classifier belonged to the action set.
        """
        return self._experience

    def __repr__(self):
        return f"{str(self.condition)} : {self.action}, F:{self.fitness}, P:{self.prediction}, E:{self.epsilon}," \
               f" N:{self.numerosity}, exp:{self.experience} "
