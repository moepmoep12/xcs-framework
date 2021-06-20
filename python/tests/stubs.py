from xcs.subsumption import ISubsumptionCriteria
from xcs.selection import IClassifierSelectionStrategy


class SubsumptionStub(ISubsumptionCriteria):
    def can_subsume(self, classifier) -> bool:
        return False


class SelectionStub(IClassifierSelectionStrategy):
    def select_classifier(self, classifier_set, score_function) -> int:
        return 0
