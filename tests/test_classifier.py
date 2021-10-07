from unittest import TestCase


class TestClassifier(TestCase):

    def test_init(self):
        from xcs_framework.xcs.classifier import Classifier
        from xcs_framework.xcs.exceptions import NoneValueException, WrongSubTypeException

        with self.assertRaises(NoneValueException):
            Classifier(None, 1)

        with self.assertRaises(WrongSubTypeException):
            Classifier('a', 1)

    def test_deep_copy(self):
        from xcs_framework.xcs.classifier import Classifier
        from xcs_framework.xcs.condition import Condition
        from xcs_framework.xcs.symbol import Symbol, WildcardSymbol
        import copy

        condition = Condition([Symbol('1'), WildcardSymbol(), Symbol('1')])
        original = Classifier(condition, 1)
        original.fitness = 10
        original.prediction = 22
        clone = copy.deepcopy(original)
        clone.epsilon = original.epsilon + 1

        self.assertTrue(original != clone)
        self.assertTrue(original.condition == clone.condition)
        self.assertTrue(original.fitness == clone.fitness)
        self.assertTrue(original.prediction == clone.prediction)
        self.assertTrue(original.epsilon != clone.epsilon)

    def test_subsumes(self):
        from xcs_framework.xcs.classifier import Classifier
        from xcs_framework.xcs.condition import Condition
        from xcs_framework.xcs.symbol import Symbol, WildcardSymbol

        condition1 = Condition([Symbol('1'), WildcardSymbol(), Symbol('1')])
        condition2 = Condition([Symbol('1'), Symbol('2'), Symbol('1')])

        action1 = 0
        action2 = 1

        cl1 = Classifier(condition1, action1)
        cl2 = Classifier(condition2, action1)
        cl3 = Classifier(condition2, action2)

        self.assertTrue(cl1.subsumes(cl2))
        self.assertFalse(cl1.subsumes(cl1))
        self.assertFalse(cl1.subsumes(cl3))
        self.assertFalse(cl2.subsumes(cl1))
        self.assertFalse(cl1.subsumes('ASD'))
