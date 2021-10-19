from unittest import TestCase

val_str: str = '42'
val_i: int = 42


class TestSymbol(TestCase):

    def test_init_symbol(self):
        from xcsframework.xcs.symbol import Symbol
        from xcsframework.xcs.exceptions import NoneValueException

        with self.assertRaises(NoneValueException):
            Symbol(None)

        s1 = Symbol(val_str)
        s2 = Symbol(val_i)

        self.assertEqual(val_str, s1.value)
        self.assertEqual(val_i, s2.value)

    def test_matches_symbol(self):
        from xcsframework.xcs.symbol import Symbol
        s1 = Symbol(val_str)

        self.assertTrue(s1.matches(val_str))
        self.assertFalse(s1.matches(val_i))

    def test_equals(self):
        from xcsframework.xcs.symbol import Symbol, WildcardSymbol, ISymbol, WILDCARD_CHAR
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

    def test_compare(self):
        from xcsframework.xcs.symbol import Symbol, WildcardSymbol, ISymbol, WILDCARD_CHAR, ComparisonResult
        from xcsframework.xcsr.center_spread.cs_symbol import CenterSpreadSymbol
        from xcsframework.xcsr.ordered_bound.ob_symbol import OrderedBoundSymbol
        from xcsframework.xcs.exceptions import NoneValueException

        s1: ISymbol = Symbol(val_str)
        s2: ISymbol = Symbol(val_str)
        s3: ISymbol = Symbol(val_i)
        s4: ISymbol = Symbol(WILDCARD_CHAR)

        w: ISymbol = WildcardSymbol()

        self.assertEqual(s1.compare(s1), ComparisonResult.EQUAL)
        self.assertEqual(s1.compare(s2), ComparisonResult.EQUAL)
        self.assertEqual(s1.compare(w), ComparisonResult.LESS_GENERAL)
        self.assertEqual(s1.compare(s3), ComparisonResult.UNDECIDABLE)
        self.assertEqual(s1.compare(s4), ComparisonResult.UNDECIDABLE)

        for item in [None, "", WILDCARD_CHAR, val_str, val_i, CenterSpreadSymbol(0, 1), OrderedBoundSymbol(0, 1)]:
            with self.assertRaises(NoneValueException):
                s1.compare(item)


class TestWildcardSymbol(TestCase):
    def test_matches(self):
        from xcsframework.xcs.symbol import WildcardSymbol

        w = WildcardSymbol()

        self.assertTrue(w.matches(val_str))
        self.assertTrue(w.matches(val_i))
        self.assertFalse(w.matches(None))

    def test_compare(self):
        from xcsframework.xcs.symbol import WildcardSymbol, ComparisonResult, Symbol, ISymbol
        from xcsframework.xcs.exceptions import NoneValueException

        w = WildcardSymbol()
        s1: ISymbol = Symbol(val_str)
        s3: ISymbol = Symbol(val_i)

        self.assertEqual(w.compare(w), ComparisonResult.EQUAL)
        self.assertEqual(w.compare(s1), ComparisonResult.MORE_GENERAL)
        self.assertEqual(w.compare(s3), ComparisonResult.MORE_GENERAL)

        with self.assertRaises(NoneValueException):
            w.compare(None)


class TestCenterSpreadSymbol(TestCase):

    def test_init(self):
        from xcsframework.xcsr.center_spread.cs_symbol import CenterSpreadSymbol
        from xcsframework.xcs.exceptions import NoneValueException, OutOfRangeException

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
        from xcsframework.xcsr.center_spread.cs_symbol import CenterSpreadSymbol

        s1 = CenterSpreadSymbol(center=val_i, spread=val_i)

        self.assertTrue(s1.matches(val_i - val_i))
        self.assertTrue(s1.matches(val_i + val_i))
        self.assertFalse(s1.matches(2 * val_i + 1))
        self.assertFalse(s1.matches(val_i - val_i - 1))

    def test_equals(self):
        from xcsframework.xcsr.center_spread.cs_symbol import CenterSpreadSymbol
        from xcsframework.xcsr.bound_symbol import BoundSymbol
        from xcsframework.xcs.symbol import ISymbol, Symbol, WildcardSymbol, WILDCARD_CHAR

        s1: BoundSymbol = CenterSpreadSymbol(center=val_i, spread=val_i)
        s2: BoundSymbol = CenterSpreadSymbol(center=val_i, spread=val_i)
        s3: ISymbol = Symbol(val_i)
        s4: ISymbol = Symbol(WILDCARD_CHAR)

        w: ISymbol = WildcardSymbol()

        self.assertTrue(s1 == s2)
        self.assertFalse(s1 == val_i)
        self.assertFalse(s1 == s3)
        self.assertFalse(s1 == w)
        self.assertFalse(s4 == w)

    def test_compare(self):
        from xcsframework.xcsr.center_spread.cs_symbol import CenterSpreadSymbol
        from xcsframework.xcsr.ordered_bound.ob_symbol import OrderedBoundSymbol
        from xcsframework.xcsr.bound_symbol import BoundSymbol
        from xcsframework.xcs.symbol import ISymbol, WildcardSymbol, ComparisonResult

        lower = 0
        upper = 10

        s1: BoundSymbol = CenterSpreadSymbol(center=val_i, spread=val_i)
        s2: BoundSymbol = CenterSpreadSymbol(center=val_i, spread=val_i + 1)
        s3: BoundSymbol = CenterSpreadSymbol(center=-val_i, spread=val_i - 1)

        o1: BoundSymbol = OrderedBoundSymbol(lower=val_i - val_i, upper=val_i + val_i)
        o2: BoundSymbol = OrderedBoundSymbol(lower=val_i - val_i + 1, upper=val_i + val_i)

        o3: BoundSymbol = OrderedBoundSymbol(lower=lower, upper=upper)
        o4: BoundSymbol = OrderedBoundSymbol(lower=lower - 2, upper=lower - 1)
        o5: BoundSymbol = OrderedBoundSymbol(lower=lower - 2, upper=lower)
        o6: BoundSymbol = OrderedBoundSymbol(lower=lower - 2, upper=lower + 1)
        o7: BoundSymbol = OrderedBoundSymbol(lower=lower - 2, upper=upper)
        o8: BoundSymbol = OrderedBoundSymbol(lower=lower - 2, upper=upper + 1)
        o9: BoundSymbol = OrderedBoundSymbol(lower=lower, upper=lower + 1)
        o10: BoundSymbol = OrderedBoundSymbol(lower=lower, upper=upper)
        o11: BoundSymbol = OrderedBoundSymbol(lower=lower, upper=upper + 1)
        o12: BoundSymbol = OrderedBoundSymbol(lower=lower + 1, upper=upper + 1)
        o13: BoundSymbol = OrderedBoundSymbol(lower=upper, upper=upper + 1)
        o14: BoundSymbol = OrderedBoundSymbol(lower=upper + 1, upper=upper + 2)

        w: ISymbol = WildcardSymbol()

        self.assertEqual(s1.compare(s1), ComparisonResult.EQUAL)

        self.assertEqual(s1.compare(o1), ComparisonResult.EQUAL)
        self.assertEqual(o1.compare(s1), ComparisonResult.EQUAL)

        self.assertEqual(s1.compare(s2), ComparisonResult.LESS_GENERAL)
        self.assertEqual(s2.compare(s1), ComparisonResult.MORE_GENERAL)

        self.assertEqual(s1.compare(o2), ComparisonResult.MORE_GENERAL)
        self.assertEqual(o2.compare(s1), ComparisonResult.LESS_GENERAL)

        self.assertEqual(s1.compare(w), ComparisonResult.LESS_GENERAL)

        self.assertEqual(s1.compare(s3), ComparisonResult.UNDECIDABLE)
        self.assertEqual(s3.compare(s1), ComparisonResult.UNDECIDABLE)

        self.assertEqual(o3.compare(o4), ComparisonResult.UNDECIDABLE)
        self.assertEqual(o4.compare(o3), ComparisonResult.UNDECIDABLE)

        self.assertEqual(o3.compare(o5), ComparisonResult.UNDECIDABLE)
        self.assertEqual(o5.compare(o3), ComparisonResult.UNDECIDABLE)

        self.assertEqual(o3.compare(o6), ComparisonResult.UNDECIDABLE)
        self.assertEqual(o6.compare(o3), ComparisonResult.UNDECIDABLE)

        self.assertEqual(o3.compare(o7), ComparisonResult.LESS_GENERAL)
        self.assertEqual(o7.compare(o3), ComparisonResult.MORE_GENERAL)

        self.assertEqual(o3.compare(o8), ComparisonResult.LESS_GENERAL)
        self.assertEqual(o8.compare(o3), ComparisonResult.MORE_GENERAL)

        self.assertEqual(o3.compare(o9), ComparisonResult.MORE_GENERAL)
        self.assertEqual(o9.compare(o3), ComparisonResult.LESS_GENERAL)

        self.assertEqual(o3.compare(o10), ComparisonResult.EQUAL)
        self.assertEqual(o10.compare(o3), ComparisonResult.EQUAL)

        self.assertEqual(o3.compare(o11), ComparisonResult.LESS_GENERAL)
        self.assertEqual(o11.compare(o3), ComparisonResult.MORE_GENERAL)

        self.assertEqual(o3.compare(o12), ComparisonResult.UNDECIDABLE)
        self.assertEqual(o12.compare(o3), ComparisonResult.UNDECIDABLE)

        self.assertEqual(o3.compare(o13), ComparisonResult.UNDECIDABLE)
        self.assertEqual(o13.compare(o3), ComparisonResult.UNDECIDABLE)

        self.assertEqual(o3.compare(o14), ComparisonResult.UNDECIDABLE)
        self.assertEqual(o14.compare(o3), ComparisonResult.UNDECIDABLE)
