from unittest import TestCase


class TestCoveringComponent(TestCase):
    def test_set_wildcard_probability(self):
        from xcs.components.covering import CoveringComponent
        from xcs.exceptions import OutOfRangeException

        covering_component = CoveringComponent(0.0)

        with self.assertRaises(OutOfRangeException):
            covering_component.wildcard_probability = -1.0

        with self.assertRaises(OutOfRangeException):
            covering_component.wildcard_probability = 'a'

        with self.assertRaises(OutOfRangeException):
            covering_component.wildcard_probability = 1.5

    def test_wildcard_probability_constructor(self):
        from xcs.components.covering import CoveringComponent
        from xcs.exceptions import OutOfRangeException

        with self.assertRaises(OutOfRangeException):
            CoveringComponent(wild_card_probability=-1.0)

        with self.assertRaises(OutOfRangeException):
            CoveringComponent(wild_card_probability='a')

        with self.assertRaises(OutOfRangeException):
            CoveringComponent(wild_card_probability=1.5)

    def test_covering_operation_no_wildcard(self):
        from xcs.components.covering import CoveringComponent
        from xcs.state import State
        covering_component = CoveringComponent(0.0)
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
        covering_component = CoveringComponent(1.0)
        available_actions = [0, 1, 2]
        state = State(['1', '0', '1'])

        result = covering_component.covering_operation(state, available_actions)

        for i, cl in enumerate(result):
            for j in range((len(cl.condition))):
                self.assertTrue(isinstance(cl.condition[j], WildcardSymbol))
            self.assertEqual(cl.action, available_actions[i])
