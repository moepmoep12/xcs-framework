import random
import sys

from xcs.components import *
from xcs.constants import *
from xcs.state import State
from xcs.subsumption import SubsumptionCriteriaExperiencePrecision
from xcs.classifier_sets import Population
from xcs.algorithm import XCS

"""
A simple example where the XCS shall learn some form of XOR (exclusive or). This is a single step problem. 
Given an input [n_0, ..., n_n] the output should be 1 if n_0 == 1 AND n_n == 0 OR n_0 == 0 AND n_n == 1.
The symbols between n_0 to n_n make the problem harder and should be ignored by the XCS.
Therefore a maximum general classifier could look like: [1 #....# 0 ] : 1.
"""


def generate_random_state(length: int):
    """
    Generates a random state filled with '0' and '1'.
    """
    s = [None] * length
    for i in range(length):
        s[i] = '0' if random.random() < 0.5 else '1'
    return State(s)


def get_expected_action(state):
    """
    :return: XOR applied on the first and last element of state.
    """
    return 1 if state[0] == '1' and state[-1] == '0' or state[0] == '0' and state[-1] == '1' else 0


def print_population(population):
    """
    Prints a population.
    """
    sorted_population = sorted(population, key=lambda cl: -cl.epsilon)
    print(f"\nPopulation with size:{population.numerosity_sum()}:")
    for cl in sorted_population:
        print(f"   {cl}")


# the length of the input. larger inputs lead to increased runtime and complexity
INPUT_LENGTH = 5

# how many learning iterations will be run
ITERATIONS = 10000

# the reward received for correct classification
MAX_REWARD = 100

# error threshold under which a classifier is considered to be accurate in its prediction
EPSILON_ZERO = sys.float_info.epsilon

# the available actions for the XCS to choose from
AVAILABLE_ACTIONS = [1, 0]

# maximum classifier count
POPULATION_SIZE = 200

# a bit higher than the default
LEARNING_RATE = 0.3

# probability for a symbol to turn into a wildcard. higher than default in this example
WILDCARD_PROBABILITY = 0.66

# required exp of a classifier to be able to subsume other classifier
MIN_EXP_SUBSUMER = 25

# size of the Test Set
TESTING_SIZE = 100

if __name__ == '__main__':
    # 1. instantiating constants (hyper parameters) and customizing values
    learning_constants = LearningConstants(epsilon_zero=EPSILON_ZERO)
    fitness_constants = FitnessConstants(alpha=0.3)
    covering_constants = CoveringConstants(wild_card_probability=WILDCARD_PROBABILITY)
    ga_constants = GAConstants()
    xcs_constants = XCSConstants()
    population_constants = PopulationConstants()

    # 2. creating xcs components
    covering_component = CoveringComponent(covering_constants=covering_constants)
    learning_component = QLearningBasedComponent(learning_constants=learning_constants,
                                                 fitness_constants=fitness_constants)
    discovery_component = GeneticAlgorithm(available_actions=AVAILABLE_ACTIONS, ga_constants=ga_constants)
    performance_component = PerformanceComponent(min_diff_actions=len(AVAILABLE_ACTIONS),
                                                 covering_component=covering_component,
                                                 available_actions=AVAILABLE_ACTIONS)

    subsumption_criteria = SubsumptionCriteriaExperiencePrecision(min_exp=MIN_EXP_SUBSUMER,
                                                                  max_epsilon=EPSILON_ZERO)

    # 3. initialize empty population
    population = Population(max_size=POPULATION_SIZE,
                            subsumption_criteria=subsumption_criteria,
                            population_constants=population_constants)

    xcs: XCS[str, int] = XCS(population=population,
                             performance_component=performance_component,
                             discovery_component=discovery_component,
                             learning_component=learning_component,
                             available_actions=AVAILABLE_ACTIONS,
                             xcs_constants=xcs_constants)

    # 4. learning
    for i in range(ITERATIONS):
        sys.stdout.write(f"\rIteration {i + 1}/{ITERATIONS}")
        state = generate_random_state(INPUT_LENGTH)
        expected_action = get_expected_action(state)
        # ~50/50 between exploration and exploitation
        chosen_action = xcs.run(state, is_explore=random.random() < 0.5)
        if chosen_action == expected_action:
            reward = MAX_REWARD
        else:
            reward = 0

        # give feedback. single step problem
        xcs.reward(reward, is_end_of_problem=True)

    # output the population
    print_population(xcs.population)

    # 5. testing
    correct = 0
    for i in range(TESTING_SIZE):
        state = generate_random_state(INPUT_LENGTH)
        expected = get_expected_action(state)
        action = xcs.query(state)
        if action == expected:
            correct += 1

    print(f"\nTesting Accuracy: {(correct / TESTING_SIZE) * 100:.1f}%")
