from unittest import TestCase


class TestSymbol(TestCase):
    val_str: str = '42'
    val_i: int = 42

    def test_init_symbol(self):
        from xcs.symbol import Symbol
        from xcs.exceptions import NoneValueException

        with self.assertRaises(NoneValueException):
            Symbol(None)

        Symbol(self.val_str)
        Symbol(self.val_i)

    def test_matches_symbol(self):
        from xcs.symbol import Symbol, ISymbol
        s1: ISymbol = Symbol(self.val_str)

        self.assertTrue(s1.matches(self.val_str))
        self.assertFalse(s1.matches(self.val_i))

    def test_matches_wildcard(self):
        from xcs.symbol import WildcardSymbol, ISymbol
        w: ISymbol = WildcardSymbol()

        self.assertTrue(w.matches(self.val_str))
        self.assertTrue(w.matches(self.val_i))

    def test_equals(self):
        from xcs.symbol import Symbol, WildcardSymbol, ISymbol, WILDCARD_CHAR
        s1: ISymbol = Symbol(self.val_str)
        s2: ISymbol = Symbol(self.val_str)
        s3: ISymbol = Symbol(self.val_i)
        s4: ISymbol = Symbol(WILDCARD_CHAR)

        w: ISymbol = WildcardSymbol()

        self.assertTrue(s1 == s2)
        self.assertTrue(s1 == self.val_str)
        self.assertFalse(s1 == s3)
        self.assertFalse(s1 == w)
        self.assertFalse(s4 == w)
