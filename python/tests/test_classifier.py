from unittest import TestCase


class TestClassifier(TestCase):
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

    def test_init(self):
        from xcs.classifier import Classifier

        with self.assertRaises(ValueError):
            Classifier(None, 1)

        with self.assertRaises(ValueError):
            Classifier('a', 1)

    def test_deep_copy(self):
        import copy
        from xcs.classifier import Classifier

        original = Classifier(self.conditions[0], 1)
        original.fitness = 10
        original.prediction = 22
        clone = copy.deepcopy(original)
        clone.epsilon = original.epsilon + 1

        self.assertTrue(original != clone)
        self.assertTrue(original.condition == clone.condition)
        self.assertTrue(original.fitness == clone.fitness)
        self.assertTrue(original.prediction == clone.prediction)
        self.assertTrue(original.epsilon != clone.epsilon)
