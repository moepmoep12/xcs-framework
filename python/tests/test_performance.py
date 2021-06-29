from unittest import TestCase
from typing import List


class TestPerformanceComponent(TestCase):
    from xcs.condition import Condition

    conditions: List[Condition[str]] = []

    @classmethod
    def setUpClass(cls) -> None:
        from xcs.condition import Condition
        from xcs.symbol import Symbol, WildcardSymbol
        cls.conditions.append(Condition([Symbol('1'), WildcardSymbol(), Symbol('1')]))
        cls.conditions.append(Condition([Symbol('0'), WildcardSymbol(), Symbol('1')]))
        cls.conditions.append(Condition([Symbol('0'), Symbol('1'), Symbol('1')]))

    def test_generate_match_set(self):
        from xcs.classifier_sets import Population, MatchSet
        from xcs.classifier import Classifier
        from xcs.components.performance import PerformanceComponent
        from xcs.state import State
        from tests.stubs import SubsumptionStub

        state: State[str] = State(['1', '0', '1'])
        cl1: Classifier[str, int] = Classifier(self.conditions[0], 1)
        cl2: Classifier[str, int] = Classifier(self.conditions[0], 0)
        cl3: Classifier[str, int] = Classifier(self.conditions[1], 0)
        population: Population[str, int] = Population(3, SubsumptionStub(), [cl1, cl2, cl3])
        performance_component = PerformanceComponent(2, None, [0, 1, 2])
        match_set: MatchSet[str, int] = performance_component.generate_match_set(population, state)

        self.assertTrue(len(match_set) == 2)
        self.assertTrue(cl1 in match_set)
        self.assertTrue(cl2 in match_set)
        self.assertTrue(cl3 not in match_set)
