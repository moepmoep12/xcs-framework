from xcs.algorithm import XCS
from xcs.components import *
from xcs.selection import RouletteWheelSelection
from xcs.state import State
from xcs.classifier_sets import Population
from xcs.subsumption import SubsumptionCriteriaExperiencePrecision
from xcs.symbol import Symbol

import random


def generate_random_input(length: int):
    state = [None] * length
    for i in range(length):
        state[i] = Symbol('0') if random.random() < 0.5 else Symbol('1')
    return State(state)


def get_expected_action(state):
    if state[0] == '1' and state[2] == '1':
        return True
    else:
        return False


if __name__ == '__main__':
    max_reward = 100
    epsilon_zero = max_reward / 100
    available_actions = [True, False]
    mutation_rate = 0.04
    fitness_reduction = 0.1
    iterations = 1000
    input_length = 5

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

    population = Population(max_size=10,
                            subsumption_criteria=subsumption_criteria)

    xcs: XCS[str, bool] = XCS(population=population,
                              performance_component=performance_component,
                              discovery_component=discovery_component,
                              learning_component=learning_component,
                              available_actions=available_actions)

    for i in range(iterations):
        state = generate_random_input(input_length)
        expected = get_expected_action(state)
        action = xcs.run(state, is_explore=True)
        if action == expected:
            reward = max_reward
        else:
            reward = -max_reward
        xcs.reward(reward, is_end_of_problem=True)

    for i in range(iterations):
        state = generate_random_input(input_length)
        expected = get_expected_action(state)
        action = xcs.run(state, is_explore=False)
        if action == expected:
            reward = max_reward
        else:
            reward = -max_reward
        xcs.reward(reward, is_end_of_problem=True)

    for i in range(20):
        state = generate_random_input(input_length)
        expected = get_expected_action(state)
        action = xcs.query(state)
        if action == expected:
            reward = max_reward
        else:
            reward = -max_reward
        print(reward)
