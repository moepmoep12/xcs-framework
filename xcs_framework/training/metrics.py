import numpy as np
from abc import abstractmethod
from overrides import overrides


class Metric:
    """
    A metric measures the value of the predicted data
    """

    @abstractmethod
    def score(self, predicted, actual):
        """
        :param predicted: The predicted classes/labels
        :param actual: The ground truth
        """
        pass


class Accuracy(Metric):
    """
    Calculates the accuracy of the prediction
    """

    @overrides
    def score(self, predicted, actual):
        num_correct_predicted = np.array([np.allclose(p, a) for p, a in zip(predicted, actual)]).sum()
        curr_accuracy = num_correct_predicted / len(predicted)
        return curr_accuracy

    def __repr__(self) -> str:
        return "Accuracy"


class Precision(Metric):
    """
    Calculates the precision of the prediction
    """

    @overrides
    def score(self, predicted, actual):
        true_positives = np.count_nonzero(predicted * actual, axis=0)
        false_positives = np.count_nonzero(predicted * (1 - actual), axis=0)
        divisor = np.maximum(true_positives + false_positives, 1E-10)
        return true_positives / divisor

    def __repr__(self) -> str:
        return "Precision"


class Recall(Metric):
    """
    Calculates the recall of the prediction
    """

    @overrides
    def score(self, predicted, actual):
        true_positives = np.count_nonzero(predicted * actual, axis=0)
        false_negatives = np.count_nonzero((1 - predicted) * actual, axis=0)
        divisor = np.maximum(true_positives + false_negatives, 1E-10)
        return true_positives / divisor

    def __repr__(self) -> str:
        return "Recall"


class F1(Metric):
    """
    Calculates the f1 score
    """

    @overrides
    def score(self, predicted, actual):
        precision = Precision().score(predicted, actual)
        recall = Recall().score(predicted, actual)
        # prevent division by 0
        divisor = np.maximum(precision + recall, 1E-10)
        return 2 * (precision * recall) / divisor

    def __repr__(self) -> str:
        return "F1"
