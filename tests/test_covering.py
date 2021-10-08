from unittest import TestCase


class TestCoveringComponent(TestCase):
    def test_covering_operation_no_wildcard(self):
        from xcsframework.xcs.components.covering import CoveringComponent
        from xcsframework.xcs.state import State
        from xcsframework.xcs.constants import CoveringConstants
        covering_component = CoveringComponent(CoveringConstants(0.0))
        available_actions = [0, 1, 2]
        state = State(['1', '0', '1'])

        result = covering_component.covering_operation(state, available_actions)

        for i, cl in enumerate(result):
            for j in range((len(cl.condition))):
                self.assertEqual(cl.condition[j].value, state[j])
            self.assertEqual(cl.action, available_actions[i])

    def test_covering_operation_all_wildcard(self):
        from xcsframework.xcs.components.covering import CoveringComponent
        from xcsframework.xcs.state import State
        from xcsframework.xcs.symbol import WildcardSymbol
        from xcsframework.xcs.constants import CoveringConstants
        covering_component = CoveringComponent(CoveringConstants(1.0))
        available_actions = [0, 1, 2]
        state = State(['1', '0', '1'])

        result = covering_component.covering_operation(state, available_actions)

        for i, cl in enumerate(result):
            for j in range((len(cl.condition))):
                self.assertTrue(isinstance(cl.condition[j], WildcardSymbol))
            self.assertEqual(cl.action, available_actions[i])


class TestCSCoveringComponent(TestCase):

    def test_init(self):
        from xcsframework.xcsr.constants import XCSRCoveringConstants
        from xcsframework.xcsr.center_spread.cs_covering import CSCoveringComponent

        max_spread = 0.5
        constants = XCSRCoveringConstants(max_spread=max_spread)
        covering_component: CSCoveringComponent = CSCoveringComponent(covering_constants=constants)

    def test_create_symbols(self):
        from xcsframework.xcsr.constants import XCSRCoveringConstants
        from xcsframework.xcsr.center_spread.cs_covering import CSCoveringComponent
        from xcsframework.xcs.exceptions import WrongSubTypeException
        from xcsframework.xcs.symbol import Symbol

        max_spread = 0.0
        constants = XCSRCoveringConstants(max_spread=max_spread)
        covering_component: CSCoveringComponent = CSCoveringComponent(covering_constants=constants)

        value = 0.5
        symbol = covering_component._create_symbol(value)

        self.assertEqual(symbol.lower_value, value)
        self.assertEqual(symbol.upper_value, value)

        with self.assertRaises(WrongSubTypeException):
            covering_component._create_symbol('v')

        with self.assertRaises(WrongSubTypeException):
            covering_component._create_symbol(Symbol('v'))
