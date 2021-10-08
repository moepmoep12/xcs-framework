from xcsframework.xcs.subsumption import ISubsumptionCriteria
from xcsframework.xcs.selection import IClassifierSelectionStrategy
from xcsframework.xcs.components.covering import ICoveringComponent, SymbolType, ActionType


class SubsumptionStub(ISubsumptionCriteria):
    def can_subsume(self, classifier) -> bool:
        return False


class SelectionStub(IClassifierSelectionStrategy):
    def select_classifier(self, classifier_set, score_function) -> int:
        return 0


class CoveringStub(ICoveringComponent):
    def covering_operation(self, current_state, available_actions):
        return []
