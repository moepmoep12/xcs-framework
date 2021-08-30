from unittest import TestCase


class TestGAConstants(TestCase):
    def test_mutation_rate(self):
        from xcs.constants import GAConstants
        ga_constants = GAConstants()
        from xcs.exceptions import OutOfRangeException

        with self.assertRaises(OutOfRangeException):
            ga_constants.mutation_rate = -1

        with self.assertRaises(OutOfRangeException):
            ga_constants.mutation_rate = 2

        with self.assertRaises(OutOfRangeException):
            ga_constants.mutation_rate = None

    def test_mutate_action(self):
        from xcs.constants import GAConstants
        ga_constants = GAConstants()
        from xcs.exceptions import WrongStrictTypeException

        with self.assertRaises(WrongStrictTypeException):
            ga_constants.mutate_action = -1

        with self.assertRaises(WrongStrictTypeException):
            ga_constants.mutate_action = '5'

        with self.assertRaises(WrongStrictTypeException):
            ga_constants.mutate_action = None

    def test_fitness_reduction(self):
        from xcs.constants import GAConstants
        ga_constants = GAConstants()
        from xcs.exceptions import OutOfRangeException

        with self.assertRaises(OutOfRangeException):
            ga_constants.fitness_reduction = -1

        with self.assertRaises(OutOfRangeException):
            ga_constants.fitness_reduction = 2

        with self.assertRaises(OutOfRangeException):
            ga_constants.fitness_reduction = None

    def test_crossover_probability(self):
        from xcs.constants import GAConstants
        ga_constants = GAConstants()
        from xcs.exceptions import OutOfRangeException

        with self.assertRaises(OutOfRangeException):
            ga_constants.crossover_probability = -1

        with self.assertRaises(OutOfRangeException):
            ga_constants.crossover_probability = 2

        with self.assertRaises(OutOfRangeException):
            ga_constants.crossover_probability = None

    def test_ga_threshold(self):
        from xcs.constants import GAConstants
        ga_constants = GAConstants()
        from xcs.exceptions import OutOfRangeException

        with self.assertRaises(OutOfRangeException):
            ga_constants.ga_threshold = -1

        with self.assertRaises(OutOfRangeException):
            ga_constants.ga_threshold = '5'

        with self.assertRaises(OutOfRangeException):
            ga_constants.ga_threshold = None

    def test_crossover_method(self):
        from xcs.constants import GAConstants
        ga_constants = GAConstants()
        from xcs.exceptions import WrongStrictTypeException

        with self.assertRaises(WrongStrictTypeException):
            ga_constants.crossover_method = -1

        with self.assertRaises(WrongStrictTypeException):
            ga_constants.crossover_method = '5'

        with self.assertRaises(WrongStrictTypeException):
            ga_constants.crossover_method = None

        ga_constants.crossover_method = GAConstants.CrossoverMethod.ONE_POINT


class TestCoveringConstants(TestCase):
    def test_wildcard_probability(self):
        from xcs.constants import CoveringConstants
        covering_constants = CoveringConstants()
        from xcs.exceptions import OutOfRangeException

        with self.assertRaises(OutOfRangeException):
            covering_constants.wildcard_probability = -1

        with self.assertRaises(OutOfRangeException):
            covering_constants.wildcard_probability = '5'

        with self.assertRaises(OutOfRangeException):
            covering_constants.wildcard_probability = None

        covering_constants.wildcard_probability = 0.5
