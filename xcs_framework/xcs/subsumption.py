from abc import ABC, abstractmethod
from overrides import overrides
from numbers import Number
from math import inf
from sys import float_info

from .exceptions import WrongSubTypeException, OutOfRangeException


class ISubsumptionCriteria(ABC):
    """
    Interface. Responsible for checking whether a classifier can subsume other classifier.
    """

    @abstractmethod
    def can_subsume(self, classifier) -> bool:
        """
        Checks whether a classifier is able to subsume other classifier.

        :param classifier: The classifier to check.
        :return: Whether the classifier can subsume other classifier.
        """
        pass


class SubsumptionCriteriaExperiencePrecision(ISubsumptionCriteria):
    """
    A classifier can subsume other classifier if it has enough experience and is precise enough.
    """

    def __init__(self, min_exp: int = 25, max_epsilon: float = float_info.epsilon):
        """
        :param min_exp: Minimum experience of a classifier required to be able to subsume other classifier.
        :param max_epsilon: The error threshold under which a classifier is considered precise enough in its prediction.
        """
        self.min_exp = min_exp
        self.max_epsilon = max_epsilon

    @property
    def min_exp(self) -> int:
        """
        :return: Minimum experience of a classifier required to be able to subsume other classifier.
        """
        return self._min_exp

    @min_exp.setter
    def min_exp(self, value: int):
        """
        :param value: Number in range [0, inf].
        : raises:
            OutOfRangeException: If value is not a number in range [0, inf].
        """
        if not isinstance(value, Number) or value < 0:
            raise OutOfRangeException(0.0, inf, value)

        self._min_exp = value

    @property
    def max_epsilon(self) -> float:
        """
        :return: The error threshold under which a classifier is considered precise in its prediction.
        """
        return self._max_epsilon

    @max_epsilon.setter
    def max_epsilon(self, value: float):
        """
        :param value: Number in range [0.0, inf].
        : raises:
            OutOfRangeException: If value is not a number in range [0.0, inf].
        """
        if not isinstance(value, Number) or value < 0.0:
            raise OutOfRangeException(0.0, inf, value)

        self._max_epsilon = value

    @overrides
    def can_subsume(self, classifier) -> bool:
        """
        raises: 
            WrongSubTypeException: If classifier is not a Classifier.
        """
        from .classifier import Classifier
        if not isinstance(classifier, Classifier):
            raise WrongSubTypeException(Classifier.__name__, type(classifier).__name__)

        return classifier.experience >= self._min_exp and classifier.epsilon <= self._max_epsilon
