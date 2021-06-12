from unittest import TestCase


class TestClassifierSet(TestCase):

    def test_set(self):
        from xcs.classifier_sets import ClassifierSet
        from xcs.classifier import Classifier
        from xcs.condition import Condition
        from xcs.symbol import Symbol, WildcardSymbol

        cond1: Condition[str] = Condition([Symbol('1'), WildcardSymbol(), Symbol('1')])
        cond2: Condition[str] = Condition([Symbol('0'), WildcardSymbol(), Symbol('1')])
        cl1: Classifier[str, int] = Classifier(condition=cond1, action=1)
        cl2: Classifier[str, int] = Classifier(condition=cond1, action=0)
        cl3: Classifier[str, int] = Classifier(condition=cond2, action=0)
        cl_set: ClassifierSet[Classifier[str, int]] = ClassifierSet([cl1, cl2])
        cl_set2: ClassifierSet[Classifier[str, int]] = ClassifierSet([cl1, cl2])
        cl_set3: ClassifierSet[Classifier[str, int]] = ClassifierSet(cl_set)
        cl_set3.append(cl3)
        available_actions = cl_set3.get_available_actions()

        self.assertTrue(len(cl_set) == 2)
        self.assertTrue(cl_set[0] == cl1)
        self.assertTrue(cl_set[1] == cl2)
        self.assertTrue(cl_set == cl_set2)
        self.assertFalse(cl_set == cl_set3)
        self.assertTrue(cl_set3[2] == cl3)
        self.assertTrue(available_actions == set([1, 0]))
