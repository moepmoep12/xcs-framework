from unittest import TestCase


class TestCondition(TestCase):
    def test_init(self):
        from xcs.condition import Condition
        from xcs.symbol import Symbol
        from xcs.exceptions import EmptyCollectionException, WrongSubTypeException

        with self.assertRaises(EmptyCollectionException):
            Condition(None)

        with self.assertRaises(EmptyCollectionException):
            Condition([])

        with self.assertRaises(WrongSubTypeException):
            Condition([Symbol('1'), 'A'])

    def test_matches(self):
        from xcs.condition import Condition
        from xcs.symbol import Symbol, WildcardSymbol
        from xcs.state import State
        state = State(['1', '0', '1'])
        c1: Condition[str] = Condition([Symbol('1'), Symbol('0'), Symbol('1')])
        c2: Condition[str] = Condition([Symbol('1'), WildcardSymbol(), Symbol('1')])
        c3: Condition[str] = Condition([Symbol('1'), Symbol('1'), Symbol('1')])
        c4: Condition[str] = Condition([Symbol(1), WildcardSymbol(), Symbol(1)])
        self.assertTrue(c1.matches(state))
        self.assertTrue(c2.matches(state))
        self.assertFalse(c3.matches(state))
        self.assertFalse(c4.matches(state))

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

    def test_is_more_general(self):
        from xcs.condition import Condition
        from xcs.symbol import Symbol, WildcardSymbol
        c1: Condition = Condition([Symbol('1'), Symbol('0'), Symbol('1')])
        c2: Condition = Condition([WildcardSymbol(), Symbol('0'), Symbol('1')])
        c3: Condition = Condition([Symbol('1'), Symbol('0'), WildcardSymbol()])
        c4: Condition = Condition([Symbol('1'), WildcardSymbol()])

        self.assertFalse(c1.is_more_general(c2))
        self.assertTrue(c2.is_more_general(c1))
        self.assertFalse(c1.is_more_general(c3))
        self.assertTrue(c3.is_more_general(c1))
        self.assertFalse(c3.is_more_general(c2))
        self.assertFalse(c2.is_more_general(c3))

        with self.assertRaises(AssertionError):
            c4.is_more_general(c1)

    def test_get_item(self):
        from xcs.condition import Condition
        from xcs.symbol import Symbol
        from xcs.exceptions import OutOfRangeException, WrongStrictTypeException
        symbol = Symbol('1')
        c: Condition = Condition([symbol, Symbol('0'), Symbol('1')])

        self.assertTrue(c[0] == symbol)

        with self.assertRaises(OutOfRangeException):
            c[-1]

        with self.assertRaises(WrongStrictTypeException):
            c['a']

    def test_set_item(self):
        from xcs.condition import Condition
        from xcs.symbol import Symbol, WildcardSymbol
        from xcs.exceptions import OutOfRangeException, WrongStrictTypeException, WrongSubTypeException
        c: Condition = Condition([Symbol('1'), Symbol('0'), Symbol('1')])
        c[0] = WildcardSymbol()

        self.assertTrue(isinstance(c[0], WildcardSymbol))

        with self.assertRaises(WrongStrictTypeException):
            c['a'] = WildcardSymbol()

        with self.assertRaises(OutOfRangeException):
            c[-1] = WildcardSymbol()
            c[3] = WildcardSymbol()

        with self.assertRaises(WrongSubTypeException):
            c[0] = 42
