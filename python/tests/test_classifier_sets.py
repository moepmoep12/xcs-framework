from unittest import TestCase


class TestClassifierSet(TestCase):
    from xcs.condition import Condition
    from typing import List

    conditions: List[Condition[str]] = []

    @classmethod
    def setUpClass(cls) -> None:
        from xcs.condition import Condition
        from xcs.symbol import Symbol, WildcardSymbol
        cls.conditions.append(Condition([Symbol('1'), WildcardSymbol(), Symbol('1')]))
        cls.conditions.append(Condition([Symbol('0'), WildcardSymbol(), Symbol('1')]))
        cls.conditions.append(Condition([Symbol('0'), Symbol('1'), Symbol('1')]))

    def test_classifier_set(self):
        from xcs.classifier_sets import ClassifierSet
        from xcs.classifier import Classifier
        from xcs.condition import Condition
        from xcs.symbol import Symbol, WildcardSymbol

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

    def test_create_population(self):
        from xcs.classifier_sets import Population
        from xcs.classifier import Classifier
        from tests.stubs import SubsumptionStub

        cl1: Classifier[str, int] = Classifier(condition=self.conditions[0], action=1)
        cl2: Classifier[str, int] = Classifier(condition=self.conditions[0], action=0)
        cl3: Classifier[str, int] = Classifier(condition=self.conditions[1], action=0)

        with self.assertRaises(AssertionError):
            Population(2, SubsumptionStub(), [cl1, cl2, cl3])

        population: Population[str, int] = Population(3, SubsumptionStub(), [cl1, cl2, cl3])

        self.assertTrue(len(population) == 3)
        self.assertTrue(population[0] == cl1)
        self.assertTrue(population[1] == cl2)
        self.assertTrue(population[2] == cl3)
