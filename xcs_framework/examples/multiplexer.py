"""
A simple example where the XCS shall learn the n-bit Multiplexer. This is a single step problem.
The example below uses the 6-bit multiplexer.
Given an input X = [n_0, ..., n_5] the output should be 0 or 1. The first two bits of X, that is n_0, n_1, serve as
address pointing to one of the remaining bits of X. For example given the input X = [0,1,1,0,1,0] the first two bits
form the binary number '01'. So the output should be X[offset + '01'] = X[2 + 1] = X[3] =  0,
where offset = 2 because we use two bits for the address.
"""

import random
import sys
from typing import List

from xcs_framework.xcs import *
from xcs_framework.training import *


class MultiplexerEnvironment(IEnvironment):
    """
    Encapsulates the Multiplexer problem in an environment.
    """

    def __init__(self, length: int, reward: Number):
        self._length = length
        self._address_length = self._get_adress_length(length, 0)
        self._reward = reward
        self._current_state = None

        # length has to be n + 2^n
        assert self._length == self._address_length + (1 << self._address_length)

    def get_state(self) -> State[int]:
        """
        Generates a random state filled with '0' and '1'.
        """
        s = [None] * self._length
        for i in range(self._length):
            s[i] = 0 if random.random() < 0.5 else 1
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
        :return: 6-bit Multiplexer applied to the state.
        """
        address = int(f"{state[0]}{state[1]}", 2)
        return state[self._address_length + address]

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


def test_data(xcs, environment, metrics, iterations):
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

# ---------------------------------------------------------------------------------------------------------------------
# TRAINING PARAMETERS
#
# after how many training iterations validation will be done
BATCH_SIZE = 300

# how many epochs will be done. total iterations = epochs * batch_size
EPOCHS = 20

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
EPSILON_ZERO = sys.float_info.epsilon

# maximum classifier count
POPULATION_SIZE = 300

# a bit higher than the default
LEARNING_RATE = 0.3

# probability for a symbol to turn into a wildcard
WILDCARD_PROBABILITY = 0.4

# fitness parameter for updating values
FITNESS_ALPHA = 0.3
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # 1. creating the environment
    environment = MultiplexerEnvironment(INPUT_LENGTH, MAX_REWARD)

    # 2. instantiating constants (hyper parameters) and customizing values
    # we only instantiate those constants of which we use different values than default
    learning_constants = LearningConstants(epsilon_zero=EPSILON_ZERO)
    fitness_constants = FitnessConstants(alpha=FITNESS_ALPHA)
    covering_constants = CoveringConstants(wild_card_probability=WILDCARD_PROBABILITY)

    # 3. creating xcs components
    covering_component = CoveringComponent(covering_constants=covering_constants)
    learning_component = QLearningBasedComponent(learning_constants=learning_constants,
                                                 fitness_constants=fitness_constants)
    discovery_component = GeneticAlgorithm(available_actions=environment.get_available_actions())
    performance_component = PerformanceComponent(min_diff_actions=len(environment.get_available_actions()),
                                                 covering_component=covering_component,
                                                 available_actions=environment.get_available_actions())

    subsumption_criteria = SubsumptionCriteriaExperiencePrecision(max_epsilon=EPSILON_ZERO)

    # 4. initialize empty population
    population = Population(max_size=POPULATION_SIZE,
                            subsumption_criteria=subsumption_criteria)

    xcs: XCS[str, int] = XCS(population=population,
                             performance_component=performance_component,
                             discovery_component=discovery_component,
                             learning_component=learning_component,
                             available_actions=environment.get_available_actions())

    # 4. initialize empty population
    population = Population(max_size=POPULATION_SIZE,
                            subsumption_criteria=subsumption_criteria)

    xcs: XCS[int, int] = XCS(population=population,
                             performance_component=performance_component,
                             discovery_component=discovery_component,
                             learning_component=learning_component,
                             available_actions=environment.get_available_actions())

    # 5.training
    trainer = TrainerEnvironment()
    accuracy_metric = Accuracy()

    print(f"Starting to learn the {INPUT_LENGTH}-bit Multiplexer...")

    for epoch in range(EPOCHS):
        trainer.optimize(xcs=xcs, environment=environment, training_iterations=BATCH_SIZE)
        # validation
        metric_scores_epoch = test_data(xcs, environment, [accuracy_metric], VALIDATION_SIZE)
        print(f"\rEpoch {epoch + 1}/{EPOCHS} --- Metrics: {metric_scores_epoch}")

    # output the population
    print_population(xcs.population, 20)

    # 6. testing
    test_accuracy = test_data(xcs, environment, [accuracy_metric], TESTING_SIZE)

    print(f"\nTesting Accuracy: {test_accuracy[0]} on {TESTING_SIZE} samples.")
