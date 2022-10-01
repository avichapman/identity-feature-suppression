from typing import List


class PerformanceData:
    """Collects and summarises classification performance data."""

    def __init__(self, field_names: List[str]):
        """Create a new classification performance data collector.

        Arguments:
            :param field_names: The name of the fields to be measured.
        """
        self.total = 0
        self.tp = 1
        self.fp = 2
        self.tn = 3
        self.fn = 4
        self.field_names = field_names
        self.field_data = []  # type: List[List[int]]
        for _ in range(len(self.field_names)):
            self.field_data.append([0, 0, 0, 0, 0])
        self.field_data.append([0, 0, 0, 0, 0])

    def add_observation(self, field_id: int, prediction: bool, truth: bool):
        """
        Adds an observation to the statistics being collected.
        :param field_id: The zero-based index of the field being recorded. Matches indices of 'field_names'.
        :param prediction: The classifiers guess about whether this flag should be 'true'.
        :param truth: The reality of whether this flag should be 'true'.
        """
        self.field_data[field_id][self.total] += 1
        self.field_data[-1][self.total] += 1
        if prediction == truth:
            if prediction:
                self.field_data[field_id][self.tp] += 1
                self.field_data[-1][self.tp] += 1
            else:
                self.field_data[field_id][self.tn] += 1
                self.field_data[-1][self.tn] += 1
        else:
            if prediction:
                self.field_data[field_id][self.fp] += 1
                self.field_data[-1][self.fp] += 1
            else:
                self.field_data[field_id][self.fn] += 1
                self.field_data[-1][self.fn] += 1

    def get_balanced_accuracy(self, field_id: int = -1) -> float:
        """
        Retrieves the balanced accuracy for a specific field.
        :argument field_id The field to get the accuracy for. If -1, gets the overall balanced accuracy.
        :returns the balanced accuracy.
        """
        values = self.field_data[field_id]
        if values[self.tp] + values[self.fn] > 0:
            tpr = values[self.tp] / (values[self.tp] + values[self.fn])
        else:
            tpr = 0.
        if values[self.tn] + values[self.fp] > 0:
            tnr = values[self.tn] / (values[self.tn] + values[self.fp])
        else:
            tnr = 0.

        balanced_accuracy = (tpr + tnr) / 2.
        return balanced_accuracy

    def get_summary_str(self) -> str:
        """
        Retrieves a string summarising the results for each field.
        :return: A multi-line string.
        """
        output = '|--------------------|-------|------|------|------|------|-----------|----------|----------|\n'
        output += '| Field              | Total |  TP  |  FP  |  TN  |  FN  | Precision |  Recall  |   bACC   |\n'
        output += '|--------------------|-------|------|------|------|------|-----------|----------|----------|\n'
        for label, values in zip(self.field_names, self.field_data[:-1]):
            if values[self.tp] + values[self.fn] > 0:
                tpr = values[self.tp] / (values[self.tp] + values[self.fn])
            else:
                tpr = 0.

            if values[self.tn] + values[self.fp] > 0:
                tnr = values[self.tn] / (values[self.tn] + values[self.fp])
            else:
                tnr = 0.

            balanced_accuracy = (tpr + tnr) / 2.
            output += '| {:18} | {:05d} | {:04d} | {:04d} | {:04d} | {:04d} | {:09f} | {:04f} | {:04f} |\n'.format(
                label,
                values[self.total],
                values[self.tp],
                values[self.fp],
                values[self.tn],
                values[self.fn],
                values[self.tp] / (values[self.tp] + values[self.fp]) if values[self.tp] + values[self.fp] > 0 else 0,
                values[self.tp] / (values[self.tp] + values[self.fn]) if values[self.tp] + values[self.fn] > 0 else 0,
                balanced_accuracy)
        output += '|--------------------|-------|------|------|------|------|-----------|----------|----------|\n'
        label = 'Overall'
        values = self.field_data[-1]
        if values[self.tp] + values[self.fn] > 0:
            tpr = values[self.tp] / (values[self.tp] + values[self.fn])
        else:
            tpr = 0.

        if values[self.tn] + values[self.fp]:
            tnr = values[self.tn] / (values[self.tn] + values[self.fp])
        else:
            tnr = 0.

        balanced_accuracy = (tpr + tnr) / 2.
        output += '| {:18} | {:05d} | {:04d} | {:04d} | {:04d} | {:04d} | {:09f} | {:04f} | {:04f} |\n'.format(
            label,
            values[self.total],
            values[self.tp],
            values[self.fp],
            values[self.tn],
            values[self.fn],
            values[self.tp] / (values[self.tp] + values[self.fp]) if values[self.tp] + values[self.fp] > 0 else 0,
            values[self.tp] / (values[self.tp] + values[self.fn]) if values[self.tp] + values[self.fn] > 0 else 0,
            balanced_accuracy)
        output += '|--------------------|-------|------|------|------|------|-----------|----------|----------|\n'
        return output

    def get_accuracy(self) -> float:
        """Accuracy"""
        total_count = self.field_data[-1][self.total]
        correct_count = self.field_data[-1][self.tp]
        if total_count > 0:
            return correct_count / total_count
        else:
            return 0.
