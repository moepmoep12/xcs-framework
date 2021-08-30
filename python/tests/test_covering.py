from unittest import TestCase


class TestCoveringComponent(TestCase):
    def test_covering_operation_no_wildcard(self):
        from xcs.components.covering import CoveringComponent
        from xcs.state import State
        from xcs.constants import CoveringConstants
        covering_component = CoveringComponent(CoveringConstants(0.0))
        available_actions = [0, 1, 2]
        state = State(['1', '0', '1'])

        result = covering_component.covering_operation(state, available_actions)

        for i, cl in enumerate(result):
            for j in range((len(cl.condition))):
                self.assertEqual(cl.condition[j].value, state[j])
            self.assertEqual(cl.action, available_actions[i])

    def test_covering_operation_all_wildcard(self):
        from xcs.components.covering import CoveringComponent
        from xcs.state import State
        from xcs.symbol import WildcardSymbol
        from xcs.constants import CoveringConstants
        covering_component = CoveringComponent(CoveringConstants(1.0))
        available_actions = [0, 1, 2]
        state = State(['1', '0', '1'])

        result = covering_component.covering_operation(state, available_actions)

        for i, cl in enumerate(result):
            for j in range((len(cl.condition))):
                self.assertTrue(isinstance(cl.condition[j], WildcardSymbol))
            self.assertEqual(cl.action, available_actions[i])
