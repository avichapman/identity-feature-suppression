from typing import Union


class Accuracy:
    """Measures accuracy."""

    def __init__(self):
        self.total_count = 0
        self.correct_count = 0
        self.truth_data = []
        self.prediction_data = []

    def update_state(self, truth: Union[bool, int], prediction: Union[bool, int]):
        """Accumulates metric statistics.
         :parameter truth: Ground Truth Value.
         :parameter prediction: The predicted value.
         """
        self.total_count += 1
        if truth == prediction:
            self.correct_count += 1

        self.truth_data.append(truth)
        self.prediction_data.append(prediction)

    def result(self) -> float:
        """Accuracy"""
        if self.total_count > 0:
            return self.correct_count / self.total_count
        else:
            return 0.
