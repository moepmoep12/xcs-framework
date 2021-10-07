from unittest import TestCase


class TestClassifierSet(TestCase):
    def test_classifier_set(self):
        from xcs_framework.xcs.classifier_sets import ClassifierSet
        from xcs_framework.xcs.classifier import Classifier
        from xcs_framework.xcs.condition import Condition
        from xcs_framework.xcs.symbol import Symbol, WildcardSymbol

        cond1: Condition[str] = Condition([Symbol('1'), WildcardSymbol(), Symbol('1')])
        cond2: Condition[str] = Condition([Symbol('0'), WildcardSymbol(), Symbol('1')])
        cl1: Classifier[str, int] = Classifier(condition=cond1, action=1)
        cl2: Classifier[str, int] = Classifier(condition=cond1, action=0)
        cl3: Classifier[str, int] = Classifier(condition=cond2, action=0)
        cl_set: ClassifierSet[str, int] = ClassifierSet([cl1, cl2])
        cl_set2: ClassifierSet[str, int] = ClassifierSet([cl1, cl2])
        cl_set3: ClassifierSet[str, int] = ClassifierSet(cl_set)
        cl_set3.insert_classifier(cl3)
        available_actions = cl_set3.get_available_actions()

        self.assertTrue(len(cl_set) == 2)
        self.assertTrue(cl_set[0] == cl1)
        self.assertTrue(cl_set[1] == cl2)
        self.assertTrue(cl_set == cl_set2)
        self.assertFalse(cl_set == cl_set3)
        self.assertTrue(cl_set3[2] == cl3)
        self.assertTrue(available_actions == set([1, 0]))

    def test_remove_classifier(self):
        from xcs_framework.xcs.classifier_sets import ClassifierSet
        from xcs_framework.xcs.classifier import Classifier
        from xcs_framework.xcs.condition import Condition
        from xcs_framework.xcs.symbol import Symbol, WildcardSymbol

        cond1: Condition[str] = Condition([Symbol('1'), WildcardSymbol(), Symbol('1')])
        cond2: Condition[str] = Condition([Symbol('0'), WildcardSymbol(), Symbol('1')])
        cl1: Classifier[str, int] = Classifier(condition=cond1, action=1)
        cl2: Classifier[str, int] = Classifier(condition=cond1, action=0)
        cl3: Classifier[str, int] = Classifier(condition=cond2, action=0)
        cl4: Classifier[str, int] = Classifier(condition=cond2, action=1)
        cl_set: ClassifierSet[str, int] = ClassifierSet([cl1, cl2, cl3])

        with self.assertRaises(ValueError):
            cl_set.remove_classifier(cl4)

        cl_set.remove_classifier(cl1)
        self.assertTrue(cl1 not in cl_set)
        self.assertTrue(cl2 in cl_set and cl3 in cl_set)

        with self.assertRaises(ValueError):
            cl_set.remove_classifier(cl1)


class TestPopulation(TestCase):

    def test_create_population(self):
        from xcs_framework.xcs.classifier_sets import Population
        from xcs_framework.xcs.classifier import Classifier
        from xcs_framework.xcs.condition import Condition
        from xcs_framework.xcs.symbol import WildcardSymbol, Symbol
        from .stubs import SubsumptionStub

        cond1 = Condition([Symbol('1'), WildcardSymbol(), Symbol('1')])
        cond2 = Condition([Symbol('0'), WildcardSymbol(), Symbol('1')])

        cl1: Classifier[str, int] = Classifier(condition=cond1, action=1)
        cl2: Classifier[str, int] = Classifier(condition=cond1, action=0)
        cl3: Classifier[str, int] = Classifier(condition=cond2, action=0)

        with self.assertRaises(AssertionError):
            Population(max_size=2, subsumption_criteria=SubsumptionStub(), classifier=[cl1, cl2, cl3])

        population: Population[str, int] = Population(max_size=3, subsumption_criteria=SubsumptionStub(),
                                                      classifier=[cl1, cl2, cl3])

        self.assertTrue(len(population) == 3)
        self.assertTrue(population[0] == cl1)
        self.assertTrue(population[1] == cl2)
        self.assertTrue(population[2] == cl3)

    def test_trim_population(self):
        from xcs_framework.xcs.classifier_sets import Population
        from xcs_framework.xcs.classifier import Classifier
        from xcs_framework.xcs.condition import Condition
        from xcs_framework.xcs.symbol import WildcardSymbol, Symbol
        from .stubs import SubsumptionStub

        cond1 = Condition([Symbol('1'), WildcardSymbol(), Symbol('1')])
        cond2 = Condition([Symbol('0'), WildcardSymbol(), Symbol('1')])

        cl1: Classifier[str, int] = Classifier(condition=cond1, action=1)
        cl1._numerosity = 3
        cl2: Classifier[str, int] = Classifier(condition=cond1, action=0)
        cl3: Classifier[str, int] = Classifier(condition=cond2, action=0)

        max_size = 3

        population: Population[str, int] = Population(max_size=max_size, subsumption_criteria=SubsumptionStub(),
                                                      classifier=[cl1, cl2, cl3])
        population2: Population[str, int] = Population(max_size=max_size, subsumption_criteria=SubsumptionStub(),
                                                       classifier=[cl1, cl2, cl3])

        population.trim_population(desired_size=population.max_size)
        self.assertEqual(population.numerosity_sum(), population.max_size)

        population.trim_population(desired_size=1)
        self.assertEqual(len(population), 1)
        self.assertEqual(population.numerosity_sum(), 1)

        with self.assertRaises(AssertionError):
            population.trim_population(max_size + 1)

        population2.trim_population(0)
        self.assertEqual(population2.numerosity_sum(), 0)
