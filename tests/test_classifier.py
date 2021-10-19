from unittest import TestCase


class TestClassifier(TestCase):

    def test_init(self):
        from xcsframework.xcs.classifier import Classifier
        from xcsframework.xcs.exceptions import NoneValueException, WrongSubTypeException

        with self.assertRaises(NoneValueException):
            Classifier(None, 1)

        with self.assertRaises(WrongSubTypeException):
            Classifier('a', 1)

    def test_deep_copy(self):
        from xcsframework.xcs.classifier import Classifier
        from xcsframework.xcs.condition import Condition
        from xcsframework.xcs.symbol import Symbol, WildcardSymbol
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
        from xcsframework.xcs.classifier import Classifier
        from xcsframework.xcs.condition import Condition
        from xcsframework.xcs.symbol import Symbol, WildcardSymbol
        from xcsframework.xcsr.center_spread.cs_symbol import CenterSpreadSymbol
        from xcsframework.xcsr.ordered_bound.ob_symbol import OrderedBoundSymbol
        from xcsframework.xcs.exceptions import NoneValueException

        condition1 = Condition([Symbol('1'), WildcardSymbol(), Symbol('1')])
        condition2 = Condition([Symbol('1'), Symbol('2'), Symbol('1')])
        condition3 = Condition([CenterSpreadSymbol(0, 1), CenterSpreadSymbol(0, 1), CenterSpreadSymbol(0, 1)])
        condition4 = Condition([OrderedBoundSymbol(-1, 1), OrderedBoundSymbol(-1, 1), OrderedBoundSymbol(-0.5, 1)])

        action1 = 0
        action2 = 1

        cl1 = Classifier(condition1, action1)
        cl2 = Classifier(condition2, action1)
        cl3 = Classifier(condition2, action2)
        cl4 = Classifier(condition3, action1)
        cl5 = Classifier(condition4, action1)

        self.assertTrue(cl1.subsumes(cl2))
        self.assertFalse(cl1.subsumes(cl1))
        self.assertFalse(cl1.subsumes(cl3))
        self.assertFalse(cl2.subsumes(cl1))
        self.assertFalse(cl1.subsumes('ASD'))
        self.assertTrue(cl4.subsumes(cl5))
        self.assertFalse(cl5.subsumes(cl4))
        with self.assertRaises(NoneValueException):
            cl1.subsumes(cl4)
