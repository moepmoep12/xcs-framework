from xcs.algorithm import XCS
from xcs.condition import Condition
from xcs.classifier import Classifier
from xcs.components import *
from xcs.selection import RouletteWheelSelection
from xcs.state import State
from xcs.classifier_sets import Population
from xcs.subsumption import SubsumptionCriteriaExperiencePrecision
from xcs.symbol import Symbol, WildcardSymbol, WILDCARD_CHAR
from xcs.constants import *

import random
import sys


def generate_random_input(length: int):
    s = [None] * length
    for i in range(length):
        s[i] = '0' if random.random() < 0.5 else '1'
    return State(s)


def get_expected_action(state):
    return 1 if state[0] == '1' and state[-1] == '0' or state[0] == '0' and state[-1] == '1' else 0


def get_symbol(char):
    return Symbol(char) if char != WILDCARD_CHAR else WildcardSymbol()


def write_kpis(xcs, accuracy_func):
    avg_accuracy = sum([accuracy_func(cl) for cl in xcs.population]) / len(population)
    avg_error = sum([cl.epsilon * cl.numerosity for cl in xcs.population]) / population.numerosity_sum()

    sys.stdout.write(
        f"Iteration:{xcs._iteration}, "
        f"Error: {avg_error:.2f},"
        f"Accuracy:{avg_accuracy:.2f} "
        f"Cl count:{len(xcs.population)},"
        f"Numerosity sum:{xcs.population.numerosity_sum()}\n")


if __name__ == '__main__':
    iterations = 20000
    max_reward = 100
    epsilon_zero = max_reward / 100
    available_actions = [1, 0]
    input_length = 5
    population_size = 200
    min_exp_subsumption = 25

    learning_constants = LearningConstants(epsilon_zero=epsilon_zero)
    fitness_constants = FitnessConstants(alpha=0.3)
    covering_constants = CoveringConstants(wild_card_probability=0.66)
    xcs_constants = XCSConstants()

    covering_component = CoveringComponent()
    learning_component = QLearningBasedComponent(learning_constants=learning_constants)

    discovery_component = GeneticAlgorithm(selection_strategy=RouletteWheelSelection(),
                                           available_actions=available_actions)

    performance_component = PerformanceComponent(min_diff_actions=len(available_actions),
                                                 covering_component=covering_component,
                                                 available_actions=available_actions)

    subsumption_criteria = SubsumptionCriteriaExperiencePrecision(min_exp=min_exp_subsumption,
                                                                  max_epsilon=epsilon_zero)

    population = Population(max_size=population_size,
                            subsumption_criteria=subsumption_criteria)

    xcs: XCS[str, int] = XCS(population=population,
                             performance_component=performance_component,
                             discovery_component=discovery_component,
                             learning_component=learning_component,
                             available_actions=available_actions)

    for i in range(iterations):
        sys.stdout.write(f"\rIteration {i + 1}/{iterations}")
        state = generate_random_input(input_length)
        expected = get_expected_action(state)
        action = xcs.run(state, is_explore=random.random() < 0.5)
        if action == expected:
            reward = max_reward
        else:
            reward = 0
        xcs.reward(reward, is_end_of_problem=True)
        # if i % 100 == 0:
        #     write_kpis(xcs, learning_component._classifier_accuracy)

    print(f"\nPopulation size: {xcs.population.numerosity_sum()}")

    sorted_population = sorted(population, key=lambda cl: -learning_component._classifier_accuracy(cl))
    for cl in sorted_population:
        print(f"{cl}, acc: {learning_component._classifier_accuracy(cl):.2f}")

    test_count = 100
    correct = 0
    for i in range(test_count):
        state = generate_random_input(input_length)
        expected = get_expected_action(state)
        action = xcs.query(state)
        if action == expected:
            correct += 1

    print(f"Accuracy: {(correct / test_count) * 100:.1f}%")
