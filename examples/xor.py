"""
A simple example where the XCS shall learn the XOR function (exclusive or). This is a single step problem.
Given an input [n_0, ..., n_n] the output should be 1 if n_0 == 1 AND n_n == 0 OR n_0 == 0 AND n_n == 1.
The symbols between n_0 to n_n make the problem harder and should be ignored by the XCS.
Therefore a maximum general classifier could look like: [1 #....# 0 ] : 1.
"""

import random
from typing import List

from xcsframework.xcs import *
from xcsframework.training import *


class XOREnvironment(IEnvironment):
    """
    Encapsulates the XOR problem in an environment.
    """

    def __init__(self, length: int, reward: Number):
        self._length = length
        self._reward = reward
        self._current_state = None

    def get_state(self) -> State[str]:
        """
        Generates a random state filled with '0' and '1'.
        """
        s = [None] * self._length
        for i in range(self._length):
            s[i] = '0' if random.random() < 0.5 else '1'
        self._current_state = State(s)
        return self._current_state

    def get_available_actions(self) -> List[int]:
        return [1, 0]

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
        :return: XOR applied on the first and last element of state.
        """
        return 1 if state[0] == '1' and state[-1] == '0' or state[0] == '0' and state[-1] == '1' else 0


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
INPUT_LENGTH = 5

# ---------------------------------------------------------------------------------------------------------------------
# TRAINING PARAMETERS
#
# after how many training iterations validation will be done
BATCH_SIZE = 500

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
EPSILON_ZERO = MAX_REWARD / 100

# maximum classifier count
POPULATION_SIZE = 200

# a bit higher than the default
LEARNING_RATE = 0.3

# probability for a symbol to turn into a wildcard. higher than default in this example
WILDCARD_PROBABILITY = 0.66

# fitness parameter for updating values
FITNESS_ALPHA = 0.3
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # 1. creating the environment
    environment = XOREnvironment(INPUT_LENGTH, MAX_REWARD)

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

    # 5.training
    trainer = TrainerEnvironment()
    accuracy_metric = Accuracy()

    print(f"Starting to learn the XOR function with input length {INPUT_LENGTH}...")

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
