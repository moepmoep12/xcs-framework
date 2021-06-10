from unittest import TestCase


class TestCondition(TestCase):
    def test_matches(self):
        from xcs.condition import Condition
        from xcs.symbol import Symbol, WildcardSymbol
        situation = ['1', '0', '1']
        c1: Condition = Condition([Symbol('1'), Symbol('0'), Symbol('1')])
        c2: Condition = Condition([Symbol('1'), WildcardSymbol(), Symbol('1')])
        c3: Condition = Condition([Symbol('1'), Symbol('1'), Symbol('1')])
        c4: Condition = Condition([Symbol(1), WildcardSymbol(), Symbol(1)])
        self.assertTrue(c1.matches(situation))
        self.assertTrue(c2.matches(situation))
        self.assertFalse(c3.matches(situation))
        self.assertFalse(c4.matches(situation))

    def test_len(self):
        from xcs.condition import Condition
        from xcs.symbol import Symbol, WildcardSymbol
        situation = ['1', '0', '1']
        c1: Condition = Condition([Symbol('1'), Symbol('0'), Symbol('1')])
        c2: Condition = Condition([WildcardSymbol(), WildcardSymbol(), WildcardSymbol()])
        self.assertEqual(len(c1), len(situation))
        self.assertEqual(len(c2), len(situation))

    def test_equal(self):
        from xcs.condition import Condition
        from xcs.symbol import Symbol, WildcardSymbol
        c1: Condition = Condition([Symbol('1'), Symbol('0'), Symbol('1')])
        c2: Condition = Condition([Symbol('1'), Symbol('0'), Symbol('1')])
        c3: Condition = Condition([WildcardSymbol(), Symbol('0'), Symbol('1')])
        c4: Condition = Condition([Symbol(1), Symbol(0), Symbol(1)])
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 == c3)
        self.assertFalse(c1 == c4)


