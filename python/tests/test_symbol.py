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
