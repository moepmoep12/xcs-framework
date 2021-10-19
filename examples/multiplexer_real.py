"""
A simple example where the XCS shall learn the 6bit Multiplexer for real values in the range [0.0, 1.0].
This is a single step problem.
Given an input X = [n_0, ..., n_n] the output should be 0 or 1. The first two bits of X, that is n_0, n_1, serve as
address pointing to one of the remaining bits of X. A threshold THETA turns a real number into an integer. So a number
0 <= x <= 1 is 0 if x < THETA, else 1.
For example given the input X = [0.4,0.7,0.8,0.2,0.8,0.1] and THETA = 0.5 the first two bits
form the binary number '01'. So the output should be X[offset + '01'] = X[2 + 1] = X[3] =  0,
where offset = 2 because we use two bits for the address.
"""

import random
import copy
from typing import List

from xcsframework.xcs import *
from xcsframework.training import *

from xcsframework.xcsr import *


class MultiplexerRealEnvironment(IEnvironment):
    """
    Encapsulates the Multiplexer problem in an environment.
    """

    def __init__(self, length: int, reward: Number, min_value: Number, max_value: Number,
                 theta: Number):
        self._length = length
        self._address_length = self._get_adress_length(length, 0)
        self._reward = reward
        self._min_value = min_value
        self._max_value = max_value
        self._theta = theta
        self._current_state = None

        # length has to be n + 2^n
        assert self._length == self._address_length + (1 << self._address_length)

    def get_state(self) -> State[float]:
        """
        Generates a random state filled with values between min_value and max_value.
        """
        s = [None] * self._length
        for i in range(self._length):
            s[i] = random.uniform(self._min_value, self._max_value)
        self._current_state = State(s)
        return self._current_state

    def get_available_actions(self) -> List[int]:
        return [0, 1]

    def execute_action(self, action: int) -> Number:
        expected = self._get_expected_action(self._current_state)
        return self._reward if expected == action else 0

    def is_end_of_problem(self) -> bool:
        """
        :return: Always true on single step problems.
        """
        return True

    def _get_expected_action(self, state) -> int:
        """
        :return: X-bit Multiplexer applied to the state.
        """
        address_string = ""
        for i in range(self._address_length):
            address_string += str(self._apply_theta(state[i]))
        address = int(address_string, self._address_length)
        return self._apply_theta(state[self._address_length + address])

    def _apply_theta(self, value: Number) -> int:
        return 0 if value < self._theta else 1

    def _get_adress_length(self, l: int, c: int):
        return c - 1 if l == 0 else self._get_adress_length(l >> 1, c + 1)


def print_population(population, amount: int = 0):
    """
    Prints a population.
    """
    sorted_population = sorted(population, key=lambda cl: -cl.experience)
    txt = ""
    if 0 < amount < population.numerosity_sum():
        txt = f" (showing {amount} most experienced classifier)"
    print(f"\nPopulation with size: {population.numerosity_sum()} {txt} :")
    for i, cl in enumerate(sorted_population):
        if 0 < amount < i:
            break
        print(f"   {cl}")


def validate(xcs, environment, metrics, iterations):
    predictions = [None] * iterations
    actual = [None] * iterations
    metric_scores = []

    for i in range(iterations):
        state = environment.get_state()
        predictions[i] = xcs.query(state)
        actual[i] = environment._get_expected_action(state)
        environment.execute_action(predictions[i])

    for metric in metrics:
        metric_scores.append((str(metric), metric.score(predictions, actual)))

    return metric_scores


# ---------------------------------------------------------------------------------------------------------------------
# PROBLEM PARAMETERS
#
# the length of the input. larger inputs lead to increased runtime and complexity
INPUT_LENGTH = 6

# threshold whether a value is 0 or 1
THETA = 0.5

# value range
MIN_VALUE = 0.0
MAX_VALUE = 1.0

# ---------------------------------------------------------------------------------------------------------------------
# TRAINING PARAMETERS
#
# how many learning iterations will be done
# after how many training iterations validation will be done
BATCH_SIZE = 200

# how many epochs will be done. total iterations = epochs * batch_size
EPOCHS = 100

# size of validation data
VALIDATION_SIZE = 50

# size of the Test Set
TESTING_SIZE = 1000

# the reward received for correct classification
MAX_REWARD = 100

# ---------------------------------------------------------------------------------------------------------------------
# XCS SPECIFIC PARAMETERS
#
# error threshold under which a classifier is considered to be accurate in its prediction
EPSILON_ZERO = MAX_REWARD / 100

# maximum classifier count
POPULATION_SIZE = 700

# probability for doing crossover in GA
CROSSOVER_PROBABILITY = 0.7

# what kind of crossover method to use in GA
CROSSOVER_METHOD = GAConstants.CrossoverMethod.UNIFORM

# whether bounds will be truncated into range [min_value, max_value]
TRUNCATE_TO_RANGE = True

# learning rate in parameter update
LEARNING_RATE = 0.03

# fitness parameter for updating values
FITNESS_ALPHA = 0.03

# mutate action in the GA. this leads to better results in this example
MUTATE_ACTION = True

# probability of mutations
MUTATION_RATE = 0.05

# how much mutation changes the value at most
MAX_MUTATION_CHANGE = THETA

# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # 1. creating the environment
    environment = MultiplexerRealEnvironment(length=INPUT_LENGTH, reward=MAX_REWARD,
                                             min_value=MIN_VALUE, max_value=MAX_VALUE, theta=THETA)

    # 2. instantiating constants (hyper parameters) and customizing values
    learning_constants = LearningConstants(beta=LEARNING_RATE, epsilon_zero=EPSILON_ZERO)
    fitness_constants = FitnessConstants(alpha=FITNESS_ALPHA)
    covering_constants = XCSRCoveringConstants(max_spread=THETA,
                                               min_value=MIN_VALUE,
                                               max_value=MAX_VALUE,
                                               truncate_to_range=TRUNCATE_TO_RANGE)
    ga_constants = GAConstants(crossover_probability=CROSSOVER_PROBABILITY,
                               mutate_action=MUTATE_ACTION,
                               mutation_rate=MUTATION_RATE,
                               crossover_method=CROSSOVER_METHOD)
    ga_constants_r = XCSRGAConstants(ga_constants=ga_constants,
                                     min_value=MIN_VALUE,
                                     max_value=MAX_VALUE,
                                     truncate_to_range=TRUNCATE_TO_RANGE,
                                     max_mutation_change=MAX_MUTATION_CHANGE)

    # 3. creating xcs components
    covering_component = CSCoveringComponent(covering_constants=covering_constants)
    learning_component = QLearningBasedComponent(learning_constants=learning_constants,
                                                 fitness_constants=fitness_constants)
    discovery_component = CSGeneticAlgorithm(available_actions=environment.get_available_actions(),
                                             ga_constants=ga_constants_r)
    performance_component = PerformanceComponent(min_diff_actions=len(environment.get_available_actions()),
                                                 covering_component=covering_component,
                                                 available_actions=environment.get_available_actions())

    subsumption_criteria = SubsumptionCriteriaExperiencePrecision(max_epsilon=EPSILON_ZERO)

    # 4. initialize empty population
    population = Population(max_size=POPULATION_SIZE,
                            subsumption_criteria=subsumption_criteria)

    xcs: XCS[float, int] = XCS(population=population,
                               performance_component=performance_component,
                               discovery_component=discovery_component,
                               learning_component=learning_component,
                               available_actions=environment.get_available_actions())

    # 5.training
    trainer = TrainerEnvironment()
    accuracy_metric = Accuracy()
    best_population = xcs.population
    best_acc = 0

    print(f"Starting to learn the {INPUT_LENGTH}-bit Multiplexer...")

    for epoch in range(EPOCHS):
        trainer.optimize(xcs=xcs, environment=environment, training_iterations=BATCH_SIZE)
        # validation
        metric_scores_epoch = validate(xcs, environment, [accuracy_metric], VALIDATION_SIZE)

        # remember best population according to validation
        accuracy = metric_scores_epoch[0][1]
        if accuracy > best_acc:
            best_acc = accuracy
            best_population = copy.deepcopy(xcs.population)

        print(f"\rEpoch {epoch + 1}/{EPOCHS} --- Metrics: {metric_scores_epoch}")

    print(f"Best average accuracy: {best_acc*100:.2f}%")

    # output the population
    xcs._population = best_population
    print_population(xcs.population, 20)

    # 6. testing
    test_accuracy = validate(xcs, environment, [accuracy_metric], TESTING_SIZE)

    print(f"\nTesting Accuracy: {test_accuracy[0]} on {TESTING_SIZE} samples.")
