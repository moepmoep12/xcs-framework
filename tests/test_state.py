from unittest import TestCase


class TestState(TestCase):
    def test_init(self):
        from xcsframework.xcs.state import State

        state = State(['1', '2', '3'])

        with self.assertRaises(TypeError):
            state[0] = 5
