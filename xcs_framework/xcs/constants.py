from numbers import Number
from math import inf
from sys import float_info
from enum import Enum

from .exceptions import OutOfRangeException, WrongStrictTypeException

"""
The constants (hyper parameters), their names and default values are based upon the paper
'An algorithmic description of XCS' by Butz & Wilson 2000 (https://doi.org/10.1007/s005000100111).
"""


class XCSConstants:
    def __init__(self,
                 gamma: Number = 0.71,
                 do_learning_subsumption: bool = True,
                 do_discovery_subsumption: bool = True,
                 subsumption_tolerance: Number = 0):
        """
        :param gamma: The discount factor of future rewards.
        :param do_learning_subsumption: Whether subsumption will be tested after learning updates.
        :param do_discovery_subsumption: Whether subsumption will be tested after rule discovery.
        :param subsumption_tolerance: The tolerance when testing for subsumption.
        """

        self.gamma = gamma
        self.do_learning_subsumption = do_learning_subsumption
        self.do_discovery_subsumption = do_discovery_subsumption
        self.subsumption_tolerance = subsumption_tolerance

    @property
    def gamma(self) -> Number:
        """
        :return: The discount factor of future rewards.
        """
        return self._gamma

    @gamma.setter
    def gamma(self, value: Number):
        """
        :param value: Float in range [0, inf].
        : raises:
            OutOfRangeException: If value is not an int in range [0, inf].
        """
        if not isinstance(value, Number) or value < 0.0:
            raise OutOfRangeException(0.0, inf, value)

        self._gamma = value

    @property
    def do_learning_subsumption(self):
        """
        :return: Whether subsumption will be tested after learning updates.
        """
        return self._do_learning_subsumption

    @do_learning_subsumption.setter
    def do_learning_subsumption(self, value: bool):
        """
        :param value: Whether subsumption will be tested after learning updates.
        :raise:
            WrongStrictTypeException: If value is not a bool.
        """
        if not isinstance(value, bool):
            raise WrongStrictTypeException(bool.__name__, type(value).__name__)

        self._do_learning_subsumption = value

    @property
    def do_discovery_subsumption(self):
        """
        :return: Whether subsumption will be tested after rule discovery.
        """
        return self._do_discovery_subsumption

    @do_discovery_subsumption.setter
    def do_discovery_subsumption(self, value: bool):
        """
        :param value: Whether subsumption will be tested after rule discovery.
        :raise:
            WrongStrictTypeException: If value is not a bool.
        """
        if not isinstance(value, bool):
            raise WrongStrictTypeException(bool.__name__, type(value).__name__)

        self._do_discovery_subsumption = value

    @property
    def subsumption_tolerance(self) -> Number:
        """
        :return: The tolerance when testing for subsumption.
        """
        return self._subsumption_tolerance

    @subsumption_tolerance.setter
    def subsumption_tolerance(self, value: Number):
        """
        :param value: The tolerance when testing for subsumption. Number > 0.
        : raises:
            OutOfRangeException: If value is not an int in range [0, inf].
        """
        if not isinstance(value, Number) or value < 0.0:
            raise OutOfRangeException(0.0, inf, value)

        self._subsumption_tolerance = value


class SymbolConstants:
    """
    Constants regarding Symbols. A Symbol encapsulates a value and can match to other values.
    """

    class SymbolRepresentation(Enum):
        """
        Representation of symbols.
        NORMAL: Use simple symbols that have a single value. A match occurs if the values are equal.
        CENTER_SPREAD: Use bound symbols with a lower and upper value. Those are defined by a center and a spread.
                       A match occurs if the given value is within the bounds.
        """
        NORMAL = 1
        CENTER_SPREAD = 2

    def __init__(self, symbol_repr: SymbolRepresentation = SymbolRepresentation.NORMAL):
        self._symbol_repr = symbol_repr

    @property
    def symbol_representation(self) -> SymbolRepresentation:
        """
        :return: Representation of symbols.
        """
        return self._symbol_repr


class ClassifierConstants:
    """
    Constants related to classifier.
    """

    def __init__(self,
                 fitness_init: Number = float_info.epsilon,
                 prediction_init: Number = float_info.epsilon,
                 epsilon_init: Number = float_info.epsilon
                 ):
        self._fitness_init = fitness_init
        self._prediction_init = prediction_init
        self._epsilon_init = epsilon_init

    @property
    def fitness_init(self) -> Number:
        """
        :return: Initial fitness value for a new classifier.
        """
        return self._fitness_init

    @property
    def prediction_init(self) -> Number:
        """
        :return: Initial prediction value for a new classifier.
        """
        return self._prediction_init

    @property
    def epsilon_init(self) -> Number:
        """
        :return: Initial epsilon value for a new classifier.
        """
        return self._epsilon_init


class PopulationConstants:
    """
    Groups constants used for the population of a XCS.
    """

    def __init__(self, theta_del: int = 25, delta: Number = 0.1):
        """
        :param theta_del: Minimum experience required for a classifier to use its fitness in deletion probability.
        :param delta: The fraction of the mean fitness of the population below which the fitness of a classifier will be
                      considered in its deletion probability.
        """
        self.theta_del = theta_del
        self.delta = delta

    @property
    def theta_del(self) -> int:
        """
        :return: Minimum experience required for a classifier to use its fitness in deletion probability.
        """
        return self._theta_del

    @theta_del.setter
    def theta_del(self, value: int):
        """
        :param value: Int in range [0, inf].
        : raises:
            OutOfRangeException: If value is not an int in range [0, inf].
        """
        if not isinstance(value, Number) or value < 0.0:
            raise OutOfRangeException(0.0, inf, value)

        self._theta_del = value

    @property
    def delta(self) -> Number:
        """
        :return: The fraction of the mean fitness of the population below which the fitness of a classifier will be
                 considered in its deletion probability. In range [0.0, inf]
        """
        return self._delta

    @delta.setter
    def delta(self, value: Number):
        """
        :param value: Float in range [0.0, inf].
        : raises:
            OutOfRangeException: If value is not a float in range [0.0, inf].
        """
        if not isinstance(value, Number) or value <= 0.0:
            raise OutOfRangeException(0.0, inf, value)

        self._delta = value


class LearningConstants:
    """
    Groups constants used for learning in a XCS.
    """

    def __init__(self, beta: Number = 0.2, epsilon_zero: Number = float_info.epsilon):
        """
        :param beta: The learning rate in range ]0.0, inf].
        :param epsilon_zero: Error threshold under which a classifier is considered 100% accurate. In range [0.0, inf].
                             Should be ~1% of maximum possible reward.
        """
        self.beta = beta
        self.epsilon_zero = epsilon_zero

    @property
    def beta(self) -> Number:
        """
        :return: The learning rate in range ]0.0, inf].
        """
        return self._beta

    @beta.setter
    def beta(self, value: Number):
        """
        :param value: Float in range ]0.0, inf].
        : raises:
            OutOfRangeException: If value is not a float in range ]0.0, inf].
        """
        if not isinstance(value, Number) or value <= 0.0:
            raise OutOfRangeException(0.0, inf, value)

        self._beta = value

    @property
    def epsilon_zero(self) -> Number:
        """
        :return: Error threshold under which a classifier is considered 100% accurate. In range [0.0, inf].
                 Should be ~1% of maximum possible reward.
        """
        return self._epsilon_zero

    @epsilon_zero.setter
    def epsilon_zero(self, value: Number):
        """
        :param value: float in range [0.0, inf].
        : raises:
            OutOfRangeException: If value is not a float in range ]0.0, inf].
        """
        if not isinstance(value, Number) or value <= 0.0:
            raise OutOfRangeException(0.0, inf, value)

        self._epsilon_zero = value


class FitnessConstants:
    """
    Groups constants used for fitness update in a XCS.
    """

    def __init__(self, alpha: Number = 0.1, nu: int = 5):
        """
        :param alpha: The learning rate for fitness updates in range ]0.0, inf].
        :param nu: The exponent for fitness updates in range ]0.0, inf].
        """
        self.alpha = alpha
        self.nu = nu

    @property
    def alpha(self) -> Number:
        """
        :return: The learning rate for fitness updates in range ]0.0, inf].
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: Number):
        """
        :param value: Float in range ]0.0, inf].
        : raises:
            OutOfRangeException: If value is not a float in range ]0.0, inf].
        """
        if not isinstance(value, Number) or value <= 0.0:
            raise OutOfRangeException(0.0, inf, value)

        self._alpha = value

    @property
    def nu(self) -> int:
        """
        :return: The exponent for fitness updates in range ]0.0, inf].
        """
        return self._nu

    @nu.setter
    def nu(self, value: int):
        """
        :param value: Int in range ]0.0, inf].
        : raises:
            OutOfRangeException: If value is not a float in range ]0.0, inf].
        """
        if not isinstance(value, Number) or value <= 0.0:
            raise OutOfRangeException(0.0, inf, value)

        self._nu = value


class GAConstants:
    """
    Groups constants used in a GA.
    """

    class CrossoverMethod(Enum):
        """
        Different methods enumerated for doing crossover in a GA.
        """
        UNIFORM = 1
        ONE_POINT = 2
        TWO_POINT = 3

    def __init__(self,
                 mutation_rate: Number = 0.03,
                 mutate_action: Number = False,
                 fitness_reduction: Number = 0.1,
                 crossover_probability: Number = 0.5,
                 ga_threshold: int = 25,
                 crossover_method: CrossoverMethod = CrossoverMethod.TWO_POINT):
        """

        :param mutation_rate: The value of the rate of mutation as a float in range [0.0, 1.0].
        :param mutate_action: Whether the action of a classifier has a chance to be mutated during discovery.
        :param fitness_reduction: Float in range [0.0, 1.0] indicating how much the fitness of a child classifier
                                  will be reduced when it is created without crossover.
        :param crossover_probability: The chance in the range [0.0, 1.0] for doing crossover in classifier discovery.
        :param ga_threshold: The minimum average time required since the last run of GA.
        :param crossover_method: Method used for doing crossover in a GA.
        """
        self.mutation_rate = mutation_rate
        self.mutate_action = mutate_action
        self.fitness_reduction = fitness_reduction
        self.crossover_probability = crossover_probability
        self.ga_threshold = ga_threshold
        self.crossover_method = crossover_method

    @property
    def mutation_rate(self) -> Number:
        """
        :return: The rate of mutation when discovering new classifier. In range [0.0, 1.0].
        """
        return self._mutation_rate

    @mutation_rate.setter
    def mutation_rate(self, value: Number):
        """
        :param value: The value of the rate of mutation as a float in range [0.0, 1.0].
        :raises:
            OutOfRangeException: If value is not a float in range [0.0, 1.0]
        """
        if not isinstance(value, Number) or value < 0.0 or value > 1.0:
            raise OutOfRangeException(0.0, 1.0, value)

        self._mutation_rate = value

    @property
    def mutate_action(self) -> bool:
        """
        :return: Whether the action will be mutated when discovering new classifier.
        """
        return self._mutate_action

    @mutate_action.setter
    def mutate_action(self, value: bool):
        """
        :param value: Whether the action of a classifier has a chance to be mutated during discovery.
        :raises:
            WrongStrictTypeException: If value is not of type bool.
        """
        if not isinstance(value, bool):
            raise WrongStrictTypeException(bool.__name__, type(value).__name__)
        self._mutate_action = value

    @property
    def fitness_reduction(self) -> Number:
        """
        :return: The percentage reduction of the fitness of a child classifier when created without crossover.
                 In range [0.0, 1.0].
        """
        return self._fitness_reduction

    @fitness_reduction.setter
    def fitness_reduction(self, value: Number):
        """
        :param value: Float in range [0.0, 1.0].
        : raises:
            OutOfRangeException: If value is not a float in range [0.0, 1.0].
        """
        if not isinstance(value, Number) or value < 0.0 or value > 1.0:
            raise OutOfRangeException(0.0, 1.0, value)

        self._fitness_reduction = value

    @property
    def crossover_probability(self) -> Number:
        """
        :return: The chance in the range [0.0, 1.0] for doing crossover in classifier discovery.
        """
        return self._crossover_probability

    @crossover_probability.setter
    def crossover_probability(self, value: Number):
        """
        :param value: Float in range [0.0, 1.0].
        : raises:
            OutOfRangeException: If value is not a float in range [0.0, 1.0].
        """
        if not isinstance(value, Number) or value < 0.0 or value > 1.0:
            raise OutOfRangeException(0.0, 1.0, value)

        self._crossover_probability = value

    @property
    def ga_threshold(self) -> int:
        """
        :return: The minimum average time required since the last run of GA. In range [0, inf].
        """
        return self._ga_threshold

    @ga_threshold.setter
    def ga_threshold(self, value: int):
        """
        :param value: Int in range [0, inf].
        : raises:
            OutOfRangeException: If value is not a int in range [0, inf].
        """
        if not isinstance(value, Number) or value < 0.0:
            raise OutOfRangeException(0, inf, value)

        self._ga_threshold = value

    @property
    def crossover_method(self) -> CrossoverMethod:
        """
        :return: The method used for doing crossover.
        """
        return self._crossover_method

    @crossover_method.setter
    def crossover_method(self, value: CrossoverMethod):
        """
        :param value: The method used for doing crossover.
        :raises:
            WrongStrictTypeException: If value is not of type CrossoverMethod.
        """
        if not isinstance(value, GAConstants.CrossoverMethod):
            raise WrongStrictTypeException(GAConstants.CrossoverMethod.__name__, type(value).__name__)

        self._crossover_method = value


class CoveringConstants:
    """
    Groups constants used in a covering component.
    """

    def __init__(self, wild_card_probability: Number = 0.33):
        """
        :param wild_card_probability: Must be number in range [0.0, 1.0].
        """
        self.wildcard_probability = wild_card_probability

    @property
    def wildcard_probability(self) -> Number:
        """
        :return: The probability for a symbol to become a wildcard.
        """
        return self._wildcard_probability

    @wildcard_probability.setter
    def wildcard_probability(self, value: Number):
        """
        :param value: The probability for a symbol to become a wildcard. Number in range [0.0, 1.0].
        :raises:
            OutOfRangeException: If wild_card_probability is not a number in range [0.0, 1.0].
        """
        if not isinstance(value, Number) or value < 0.0 or value > 1.0:
            raise OutOfRangeException(0.0, 1.0, value)

        self._wildcard_probability = value
