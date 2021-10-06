from unittest import TestCase

val_str: str = '42'
val_i: int = 42


class TestSymbol(TestCase):

    def test_init_symbol(self):
        from xcs.symbol import Symbol
        from xcs.exceptions import NoneValueException

        with self.assertRaises(NoneValueException):
            Symbol(None)

        s1 = Symbol(val_str)
        s2 = Symbol(val_i)

        self.assertEqual(val_str, s1.value)
        self.assertEqual(val_i, s2.value)

    def test_matches_symbol(self):
        from xcs.symbol import Symbol
        s1 = Symbol(val_str)

        self.assertTrue(s1.matches(val_str))
        self.assertFalse(s1.matches(val_i))

    def test_equals(self):
        from xcs.symbol import Symbol, WildcardSymbol, ISymbol, WILDCARD_CHAR
        s1: ISymbol = Symbol(val_str)
        s2: ISymbol = Symbol(val_str)
        s3: ISymbol = Symbol(val_i)
        s4: ISymbol = Symbol(WILDCARD_CHAR)

        w: ISymbol = WildcardSymbol()

        self.assertTrue(s1 == s2)
        self.assertTrue(s1 == val_str)
        self.assertFalse(s1 == s3)
        self.assertFalse(s1 == w)
        self.assertFalse(s4 == w)


class TestWildcardSymbol(TestCase):
    def test_matches(self):
        from xcs.symbol import WildcardSymbol
        w = WildcardSymbol()

        self.assertTrue(w.matches(val_str))
        self.assertTrue(w.matches(val_i))


class TestCenterSpreadSymbol(TestCase):

    def test_init(self):
        from xcs.symbol import CenterSpreadSymbol
        from xcs.exceptions import NoneValueException, OutOfRangeException

        with self.assertRaises(NoneValueException):
            CenterSpreadSymbol(center=None, spread=1)
        with self.assertRaises(NoneValueException):
            CenterSpreadSymbol(center=1, spread=None)

        with self.assertRaises(OutOfRangeException):
            CenterSpreadSymbol(center=0.5, spread=-1)

        center = 0.5
        spread = 0.5

        s1 = CenterSpreadSymbol(center=center, spread=spread)

        self.assertEqual(center - spread, s1.lower_value)
        self.assertEqual(center + spread, s1.upper_value)

    def test_matches(self):
        from xcs.symbol import CenterSpreadSymbol
        s1 = CenterSpreadSymbol(center=val_i, spread=val_i)

        self.assertTrue(s1.matches(val_i - val_i))
        self.assertTrue(s1.matches(val_i + val_i))
        self.assertFalse(s1.matches(2 * val_i + 1))
        self.assertFalse(s1.matches(val_i - val_i - 1))

    def test_equals(self):
        from xcs.symbol import Symbol, WildcardSymbol, CenterSpreadSymbol, ISymbol, WILDCARD_CHAR
        s1: ISymbol = CenterSpreadSymbol(center=val_i, spread=val_i)
        s2: ISymbol = CenterSpreadSymbol(center=val_i, spread=val_i)
        s3: ISymbol = Symbol(val_i)
        s4: ISymbol = Symbol(WILDCARD_CHAR)

        w: ISymbol = WildcardSymbol()

        self.assertTrue(s1 == s2)
        self.assertFalse(s1 == val_i)
        self.assertFalse(s1 == s3)
        self.assertFalse(s1 == w)
        self.assertFalse(s4 == w)
