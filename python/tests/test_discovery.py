from unittest import TestCase


class TestGeneticAlgorithm(TestCase):

    def test__generate_child(self):
        from xcs.condition import Condition
        from xcs.symbol import Symbol, WildcardSymbol
        from xcs.classifier import Classifier
        from xcs.components.discovery import GeneticAlgorithm
        from tests.stubs import SelectionStub
        ga = GeneticAlgorithm(SelectionStub(), [0])
        condition: Condition[str] = Condition([Symbol('1'), WildcardSymbol(), Symbol('1')])
        action: int = 1
        parent: Classifier[str, int] = Classifier(condition, action)
        parent.fitness = 100
        parent.prediction = 50
        parent.epsilon = 10
        parent._experience = 22
        parent._numerosity = 10
        child: Classifier[str, int] = ga._generate_child(parent)

        self.assertNotEqual(parent, child)
        self.assertEqual(parent.condition, child.condition)
        self.assertEqual(parent.action, child.action)
        self.assertEqual(parent.prediction, child.prediction)
        self.assertEqual(parent.epsilon, child.epsilon)
        self.assertNotEqual(parent.experience, child.experience)
        self.assertNotEqual(parent.numerosity, child.numerosity)
        self.assertNotEqual(parent.fitness, child.fitness)

    def test__swap_symbols_one_element(self):
        from xcs.condition import Condition
        from xcs.symbol import Symbol, WildcardSymbol
        from xcs.components.discovery import GeneticAlgorithm
        from tests.stubs import SelectionStub
        import copy
        ga = GeneticAlgorithm(SelectionStub(), [0])
        symbols1 = [Symbol('1'), WildcardSymbol(), Symbol('1')]
        symbols2 = [Symbol('0'), Symbol('0'), Symbol('0')]

        # swap n-th element only
        for i in range(len(symbols1)):
            condition1: Condition[str] = Condition(copy.deepcopy(symbols1))
            condition2: Condition[str] = Condition(copy.deepcopy(symbols2))
            from_index = to_index = i
            swapped = ga._swap_symbols(condition1, condition2, from_index, to_index)
            self.assertTrue(swapped)
            for j in range(len(condition1)):
                if j != i:
                    self.assertEqual(condition1[j], symbols1[j])
                    self.assertEqual(condition2[j], symbols2[j])
                else:
                    self.assertEqual(condition1[j], symbols2[j])
                    self.assertEqual(condition2[j], symbols1[j])

    def test_swap_symbols_multiple_elements(self):
        from xcs.condition import Condition
        from xcs.symbol import Symbol, WildcardSymbol
        from xcs.components.discovery import GeneticAlgorithm
        from tests.stubs import SelectionStub
        import copy
        ga = GeneticAlgorithm(SelectionStub(), [0])
        symbols1 = [Symbol('1'), WildcardSymbol(), Symbol('1')]
        symbols2 = [Symbol('0'), Symbol('0'), Symbol('0')]
        condition1: Condition[str] = Condition(copy.deepcopy(symbols1))
        condition2: Condition[str] = Condition(copy.deepcopy(symbols2))

        from_index = 0
        to_index = 2
        swapped = ga._swap_symbols(condition1, condition2, from_index, to_index)

        self.assertTrue(swapped)
        for i in range(len(condition1)):
            self.assertEqual(condition1[i], symbols2[i])
            self.assertEqual(condition2[i], symbols1[i])

    def test__swap_symbols_exception(self):
        from xcs.condition import Condition
        from xcs.symbol import Symbol, WildcardSymbol
        from xcs.components.discovery import GeneticAlgorithm
        from xcs.exceptions import NoneValueException, EmptyCollectionException, OutOfRangeException
        from tests.stubs import SelectionStub
        import copy
        ga = GeneticAlgorithm(SelectionStub(), [0])
        symbols1 = [Symbol('1'), WildcardSymbol(), Symbol('1')]
        symbols2 = [Symbol('0'), Symbol('0')]
        condition0: Condition[str] = Condition([Symbol('1')])
        condition0._condition = []
        condition1: Condition[str] = Condition(symbols1)
        condition2: Condition[str] = Condition(symbols2)
        condition3: Condition[str] = Condition(copy.deepcopy(symbols1))
        condition4: Condition[str] = condition1

        # None condition
        with self.assertRaises(NoneValueException):
            ga._swap_symbols(None, condition1, 0, 0)

        # empty condition
        with self.assertRaises(EmptyCollectionException):
            ga._swap_symbols(condition0, condition1, 0, 0)

        # different length
        with self.assertRaises(ValueError):
            ga._swap_symbols(condition1, condition2, 0, 0)

        # swapping with itself
        with self.assertRaises(ValueError):
            ga._swap_symbols(condition1, condition4, 0, 0)

        # invalid from_index
        with self.assertRaises(OutOfRangeException):
            ga._swap_symbols(condition1, condition3, -1, 0)
            ga._swap_symbols(condition1, condition3, 3, 0)

        # invalid to_index
        with self.assertRaises(OutOfRangeException):
            ga._swap_symbols(condition1, condition3, 0, -1)
            ga._swap_symbols(condition1, condition3, 2, 3)

        # invalid range
        with self.assertRaises(ValueError):
            ga._swap_symbols(condition1, condition3, 2, 1)

    def test_mutation_rate_setter(self):
        from xcs.components.discovery import GeneticAlgorithm
        from tests.stubs import SelectionStub
        ga = GeneticAlgorithm(SelectionStub(), [0])
        from xcs.exceptions import OutOfRangeException

        with self.assertRaises(OutOfRangeException):
            ga.mutation_rate = -1

        with self.assertRaises(OutOfRangeException):
            ga.mutation_rate = 2

        with self.assertRaises(OutOfRangeException):
            ga.mutation_rate = None

    def test_crossover_probability_setter(self):
        from xcs.components.discovery import GeneticAlgorithm
        from tests.stubs import SelectionStub
        ga = GeneticAlgorithm(SelectionStub(), [0])
        from xcs.exceptions import OutOfRangeException

        with self.assertRaises(OutOfRangeException):
            ga.crossover_probability = -1

        with self.assertRaises(OutOfRangeException):
            ga.crossover_probability = 2

        with self.assertRaises(OutOfRangeException):
            ga.crossover_probability = None

    def test_fitness_reduction_setter(self):
        from xcs.components.discovery import GeneticAlgorithm
        from tests.stubs import SelectionStub
        ga = GeneticAlgorithm(SelectionStub(), [0])
        from xcs.exceptions import OutOfRangeException

        with self.assertRaises(OutOfRangeException):
            ga.fitness_reduction = -1

        with self.assertRaises(OutOfRangeException):
            ga.fitness_reduction = 2

        with self.assertRaises(OutOfRangeException):
            ga.fitness_reduction = None

    def test_selection_strategy_setter(self):
        from xcs.components.discovery import GeneticAlgorithm
        from tests.stubs import SelectionStub
        ga = GeneticAlgorithm(SelectionStub(), [0])
        from xcs.exceptions import WrongSubTypeException

        with self.assertRaises(WrongSubTypeException):
            ga.selection_strategy = 0

        with self.assertRaises(WrongSubTypeException):
            ga.selection_strategy = 'a'

        with self.assertRaises(WrongSubTypeException):
            ga.selection_strategy = None
