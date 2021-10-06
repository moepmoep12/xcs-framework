from unittest import TestCase


class TestSubsumptionCriteriaExperiencePrecision(TestCase):
    def test_min_exp(self):
        from xcs.subsumption import SubsumptionCriteriaExperiencePrecision
        from xcs.exceptions import OutOfRangeException
        sub_criteria = SubsumptionCriteriaExperiencePrecision(0, 0)
        sub_criteria.min_exp = 2

        self.assertEqual(sub_criteria.min_exp, 2)

        with self.assertRaises(OutOfRangeException):
            sub_criteria.min_exp = -1

        with self.assertRaises(OutOfRangeException):
            sub_criteria.min_exp = None

        with self.assertRaises(OutOfRangeException):
            sub_criteria.min_exp = 'a'

    def test_max_epsilon(self):
        from xcs.subsumption import SubsumptionCriteriaExperiencePrecision
        from xcs.exceptions import OutOfRangeException
        sub_criteria = SubsumptionCriteriaExperiencePrecision(0, 0)
        sub_criteria.max_epsilon = 1.0

        self.assertEqual(sub_criteria.max_epsilon, 1.0)

        with self.assertRaises(OutOfRangeException):
            sub_criteria.max_epsilon = -1

        with self.assertRaises(OutOfRangeException):
            sub_criteria.max_epsilon = None

        with self.assertRaises(OutOfRangeException):
            sub_criteria.max_epsilon = 'a'

    def test_can_subsume(self):
        from xcs.subsumption import SubsumptionCriteriaExperiencePrecision
        from xcs.condition import Condition
        from xcs.classifier import Classifier
        from xcs.symbol import WildcardSymbol, Symbol
        from xcs.exceptions import WrongSubTypeException
        min_exp = 10
        max_epsilon = 5.0
        sub_criteria = SubsumptionCriteriaExperiencePrecision(min_exp=min_exp, max_epsilon=max_epsilon)
        condition1 = Condition([Symbol('1'), WildcardSymbol(), Symbol('1')])
        condition2 = Condition([Symbol('1'), WildcardSymbol(), WildcardSymbol()])
        cl1 = Classifier(condition1, 1)
        cl1._experience = min_exp
        cl1._epsilon = max_epsilon
        cl2 = Classifier(condition2, 1)

        self.assertTrue(sub_criteria.can_subsume(cl1))
        self.assertFalse(sub_criteria.can_subsume(cl2))

        with self.assertRaises(WrongSubTypeException):
            sub_criteria.can_subsume(None)
