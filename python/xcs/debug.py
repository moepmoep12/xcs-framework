from xcs.algorithm import XCS
from sys import float_info
from xcs.condition import Condition
from xcs.classifier import Classifier
from xcs.components import *
from xcs.selection import RouletteWheelSelection
from xcs.state import State
from xcs.classifier_sets import Population
from xcs.subsumption import SubsumptionCriteriaExperiencePrecision
from xcs.symbol import Symbol, WildcardSymbol, WILDCARD_CHAR

import random


def generate_random_input(length: int):
    state = [None] * length
    for i in range(length):
        state[i] = Symbol('0') if random.random() < 0.5 else Symbol('1')
    return State(state)


def get_expected_action(state):
    return 1 if state[0] == '1' else 0


def get_symbol(char):
    return Symbol(char) if char != WILDCARD_CHAR else WildcardSymbol()


def create_classifier(condition, action, f, p, e, n, exp, a):
    cond = Condition([get_symbol(c) for c in condition])
    return Classifier(condition=cond, action=action, f=f, p=p, e=e, n=n, exp=exp, a=a)


if __name__ == '__main__':
    max_reward = 100
    epsilon_zero = max_reward / 100
    available_actions = [1, 0]
    mutation_rate = 0.04
    fitness_reduction = 0.1
    iterations = 20000
    input_length = 5
    population_size = 25

    covering_component = CoveringComponent(wild_card_probability=0.33)

    learning_component = QLearningBasedComponent(learning_rate_prediction=0.2)
    learning_component._epsilon_zero = epsilon_zero
    learning_component._learning_rate_fitness = 0.1

    discovery_component = GeneticAlgorithm(selection_strategy=RouletteWheelSelection(),
                                           available_actions=available_actions)

    performance_component = PerformanceComponent(min_diff_actions=len(available_actions),
                                                 covering_component=covering_component,
                                                 available_actions=available_actions)

    subsumption_criteria = SubsumptionCriteriaExperiencePrecision(min_exp=20,
                                                                  max_epsilon=epsilon_zero)

    population = Population(max_size=population_size,
                            subsumption_criteria=subsumption_criteria)

    xcs: XCS[str, bool] = XCS(population=population,
                              performance_component=performance_component,
                              discovery_component=discovery_component,
                              learning_component=learning_component,
                              available_actions=available_actions)

    for i in range(iterations):
        state = generate_random_input(input_length)
        expected = get_expected_action(state)
        action = xcs.run(state, is_explore=random.random() < 0.5)
        if action == expected:
            reward = max_reward
        else:
            reward = -max_reward
        xcs.reward(reward, is_end_of_problem=True)

    # for i in range(iterations):
    #     state = generate_random_input(input_length)
    #     expected = get_expected_action(state)
    #     action = xcs.run(state, is_explore=False)
    #     if action == expected:
    #         reward = max_reward
    #     else:
    #         reward = -max_reward
    #     xcs.reward(reward, is_end_of_problem=True)

    print(f"Population size: {xcs.population.numerosity_sum()}")

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

    print(f"Accuracy: {(correct / test_count)*100:.1f}%")


    # pop = []
    # pop.append(create_classifier('#####', 1, 1.0, 100.0, float_info.epsilon, 10, 525, 10.0))
    # pop.append(create_classifier('0####', 0, 0.995, 100.0, float_info.epsilon, 18, 524, 18.094))
    # pop.append(create_classifier('##11#', 0, 0.053, 100.0, float_info.epsilon, 1, 139, 19.0))
    # pop.append(create_classifier('0####', 1, 0.989, -100.0, float_info.epsilon, 8, 118, 7.977))
    # pop.append(create_classifier('1####', 0, 0.997, -100.0, float_info.epsilon, 13, 51, 14.269))
    #
    # population = Population(max_size=population_size,
    #                         subsumption_criteria=subsumption_criteria,
    #                         classifier=pop)
    #
    # xcs: XCS[str, bool] = XCS(population=population,
    #                           performance_component=performance_component,
    #                           discovery_component=discovery_component,
    #                           learning_component=learning_component,
    #                           available_actions=available_actions)
    #
    # for i in range(iterations):
    #     state = generate_random_input(input_length)
    #     expected = get_expected_action(state)
    #     action = xcs.run(state, is_explore=True)
    #     if action == expected:
    #         reward = max_reward
    #     else:
    #         reward = -max_reward
    #     xcs.reward(reward, is_end_of_problem=True)
