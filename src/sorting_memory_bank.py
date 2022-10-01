from typing import Tuple, List

import torch


class SortingMemoryBank:
    """
    Keeps track of metrics associated with every sample in a dataset, along with their labels and whether the labels
    are incorrect. Allows for calculation of probabilities of labels being incorrect.
    """

    def __init__(self,
                 dataset_size: int,
                 use_cuda: bool = None,
                 history_length: int = 1,
                 history_forward: bool = False):
        """
        Creates a new memory bank.
        :param dataset_size: The total number of samples in the dataset.
        :param use_cuda: If true and available, store in CUDA memory.
        :param history_length: The number of previous epochs of sample history to store.
        :param history_forward: If true, save history until we reach the length and then stop saving new history.
        If false, always keep the latest X epochs' worth of data and forget anything older.
        """
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()

        self.dataset_size = dataset_size
        self.use_cuda = use_cuda
        self.history_length = history_length
        self.history_forward = history_forward

        # Each element in these buffers corresponds to a sample in the dataset. The number stored in the buffer
        # corresponds to the position in an ascending or descending list of sample values.
        #
        # e.g. If sample 0 has the tenth lowest measurement, the value at that position '0' will be '9'.
        self.sample_measurement_ordering_ascending = torch.tensor([-1] * self.dataset_size)
        self.sample_measurement_ordering_descending = torch.tensor([-1] * self.dataset_size)

        # The actual measurement for sample in the dataset...
        self.sample_metric: List[torch.Tensor] = []
        self.sample_metric_epoch: List[int] = []
        self.sample_metric_mean = torch.tensor([-1.] * self.dataset_size)

        # The apparent label for each sample in the dataset...
        self.sample_labels = torch.tensor([-1] * self.dataset_size)

        # The true label for each sample in the dataset...
        self.sample_true_labels = torch.tensor([-1] * self.dataset_size)

        if use_cuda:
            self.sample_measurement_ordering_ascending = self.sample_measurement_ordering_ascending.cuda()
            self.sample_measurement_ordering_descending = self.sample_measurement_ordering_descending.cuda()
            self.sample_metric_mean = self.sample_metric_mean.cuda()
            self.sample_labels = self.sample_labels.cuda()
            self.sample_true_labels = self.sample_true_labels.cuda()

    def not_accepting_new_data(self, epoch: int):
        """
        Returns true if the history is full and we are not accepting new data now.
        :param epoch The current epoch.
        """
        if len(self.sample_metric) == 0:
            # No history, it's ok to add some...
            return False

        latest_history_is_for_older_epoch = epoch > self.sample_metric_epoch[-1]
        no_room_for_more_history = len(self.sample_metric) >= self.history_length
        return no_room_for_more_history and self.history_forward and latest_history_is_for_older_epoch

    def start_new_epoch(self, epoch: int):
        """
        This should be called at the start of every epoch being recorded. It will set up the data recording for that
        epoch.
        :argument epoch The new epoch's number.
        """
        # Check if the history is full...
        if self.not_accepting_new_data(epoch):
            # We have already filled up the history. No more...
            return

        if len(self.sample_metric) == self.history_length:
            # Remove the first (and therefore oldest) element in the list...
            self.sample_metric.pop(0)
            self.sample_metric_epoch.pop(0)

        # Now set up new epoch...
        self.sample_metric.append(torch.tensor([-1.] * self.dataset_size))
        self.sample_metric_epoch.append(epoch)

        if self.use_cuda:
            self.sample_metric[-1] = self.sample_metric[-1].cuda()

    def add_to_memory(self,
                      epoch: int,
                      sample_indices: torch.Tensor,
                      labels: torch.Tensor,
                      measurement: torch.Tensor,
                      true_labels: torch.Tensor):
        """
        Adds a measurement to memory, along with the apparent truth label.
        :argument epoch The current epoch.
        :argument sample_indices The indices of the samples in the training batch.
        :argument labels The apparent truth labels for the samples in the training batch.
        :argument measurement The data being measured for each element in the batch, such as loss or entropy.
        :argument true_labels The true labels for collecting measurements vs truth statistics.
        """
        if self.not_accepting_new_data(epoch):
            return

        assert sample_indices.size()[0] == labels.size()[0] == len(measurement), \
            "Indices, logits and labels must be the same size."

        self.sample_labels[sample_indices.detach()] = labels.detach()
        self.sample_metric[-1][sample_indices.detach()] = measurement.detach()
        self.sample_true_labels[sample_indices.detach()] = true_labels.detach()

    def is_in_percentile_range(self, start: float, stop: float, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Given the indices of the samples in the training batch, returns a selection tensor for the batch elements
        whose recent measurements place it within the provided percentile range.
        :argument start The lowest percentile to select. Min value is 0.
        :argument stop The highest percentile to select. Max value is 100.
        :argument batch_indices The indices of the samples in the training mini-batch.
        :returns Boolean tensor of the same length as `batch_indices` with true for elements that are in that bottom
        set.
        """
        highest_value = int((stop / 100.) * (self.dataset_size - 1))
        lowest_value = int((start / 100.) * (self.dataset_size - 1))
        not_too_low = self.sample_measurement_ordering_ascending >= lowest_value
        not_too_high = self.sample_measurement_ordering_ascending <= highest_value
        in_range = torch.logical_and(not_too_low, not_too_high)
        batch_in_range = in_range[batch_indices]
        return batch_in_range

    def is_in_bottom_x(self, batch_indices: torch.Tensor, number: int) -> torch.Tensor:
        """
        Given the indices of the samples in the training batch, returns a selection tensor for the batch elements
        whose recent measurements place it in the bottom `number` for their label.
        For example, calling `is_in_bottom_x(batch_indices, 1)` will return a selection tensor with a True for every
        element that had the lowest for its label across the entire training dataset.
        :argument batch_indices The indices of the samples in the training mini-batch.
        :argument number 1 if we want the lowest measurement, 5 if we want in the bottom five, etc.
        :returns Boolean tensor of the same length as `batch_indices` with true for elements that are in that bottom
        set.
        """
        in_bottom = self.sample_measurement_ordering_ascending < number
        batch_in_bottom = in_bottom[batch_indices]
        return batch_in_bottom

    def is_in_top_x(self, batch_indices: torch.Tensor, number: int) -> torch.Tensor:
        """
        Given the indices of the samples in the training batch, returns a selection tensor for the batch elements
        whose recent measurements place it in the top `number` for their label.
        For example, calling `is_in_top_x(batch_indices, 1)` will return a selection tensor with a True for every
        element that had the highest loss for its label across the entire training dataset.
        :argument batch_indices The indices of the samples in the training mini-batch.
        :argument number 1 if we want the lowest measurement, 5 if we want in the bottom five, etc.
        :returns Boolean tensor of the same length as `batch_indices` with true for elements that are in that bottom
        set.
        """
        in_top = self.sample_measurement_ordering_descending < number
        batch_in_top = in_top[batch_indices]
        return batch_in_top

    def normalise_latest_measurements(self):
        """
        Normalise the measurements so that their mean is zero and their standard deviation is one.
        """
        mean = self.sample_metric[-1].mean(dim=0).item()
        std_deviation = self.sample_metric[-1].std(dim=0).item()
        self.sample_metric[-1] = (self.sample_metric[-1] - mean) / std_deviation

        self._calculate_average_over_history()

    def normalise_latest_measurements_by_label(self):
        """
        Normalise the measurements so that for each apparent label the mean is zero and their standard deviation is one.
        """
        min_label = self.sample_labels.min().item()
        max_label = self.sample_labels.max().item()
        for label in range(min_label, max_label + 1):
            indices = self.sample_labels == label
            if indices.sum().item() > 0:
                mean = self.sample_metric[-1][indices].mean(dim=0).item()
                std_deviation = self.sample_metric[-1][indices].std(dim=0).item()
                self.sample_metric[-1][indices] = (self.sample_metric[-1][indices] - mean) / std_deviation

        self._calculate_average_over_history()

    def build_lowest_first_measurement_cache(self):
        """
        Once an epoch, this should be called to rebuild the measurement ordering cache.
        The measurement ordering cache will be ordered such that the lowest indices correspond to the lowest
        measurements.
        """
        self._build_measurement_cache(buffer=self.sample_measurement_ordering_ascending, reverse=False)

    def build_highest_first_measurement_cache(self):
        """
        Once an epoch, this should be called to rebuild the measurement ordering cache.
        The measurement ordering cache will be ordered such that the lowest indices correspond to the highest
        measurements.
        """
        self._build_measurement_cache(buffer=self.sample_measurement_ordering_descending, reverse=True)

    def get_histogram_data(self, bin_size: int = 500, ascending: bool = True, filter_to_label: int = None) -> \
            Tuple[List[float], List[float], List[float]]:
        """
        Divides the dataset into bins and returns three lists with the average measurements over history and the number
        of dirty and total labels in each bin, sorted by measurement in either ascending or descending order.
        :param bin_size: The desired size for each bin. Defaults to 500, which is 1% of the CIFAR10 dataset.
        :param ascending: If true, sort in ascending order.
        :param filter_to_label: If provided, only provide data for a given label.
        :return: A tuple of three lists:
            - The average measurement value for each of 100 bins.
            - The number of incorrect labels for each of 100 bins.
            - The number of total labels for each of 100 bins.
        """
        items = []
        for i in range(self.sample_metric_mean.size()[0]):
            sample_label = self.sample_labels[i].item()
            if (filter_to_label is None or sample_label == filter_to_label) and sample_label >= 0:
                items.append((i, self.sample_metric_mean[i].item()))
        items = sorted(items, key=lambda datum: datum[1], reverse=not ascending)

        measurement_bins = []
        incorrect_label_bins = []
        total_label_bins = []
        for position in range(0, len(items), bin_size):
            measurement = 0.
            label_count = 0
            false_label_count = 0
            for sub_pos in range(position, position + bin_size, 1):
                if sub_pos >= len(items):
                    break

                index = items[sub_pos][0]
                sample_label = self.sample_labels[index].item()
                if (filter_to_label is None or sample_label == filter_to_label) and sample_label >= 0:
                    measurement += self.sample_metric_mean[index].item()
                    label_count += 1

                    if self.sample_labels[index].item() != self.sample_true_labels[index].item():
                        false_label_count += 1

            measurement_bins.append(measurement / label_count)
            total_label_bins.append(label_count)
            incorrect_label_bins.append(false_label_count)

        return measurement_bins, incorrect_label_bins, total_label_bins

    def _calculate_average_over_history(self):
        """
        Takes the sample metric over the history and calculates the average for every individual sample.
        """
        self.sample_metric_mean = torch.stack(self.sample_metric).mean(dim=0)

    def _build_measurement_cache(self, buffer: torch.Tensor, reverse: bool):
        """
        Once an epoch, this should be called to rebuild the measurement ordering cache.
        :argument buffer: The place to store the indices.
        :argument reverse: If true, build the measurement cache so that the lowest values represent the highest
        measurement.
        """
        items = []
        for i in range(self.sample_metric_mean.size()[0]):
            items.append((i, self.sample_metric_mean[i].item()))

        items = sorted(items, key=lambda datum: datum[1], reverse=reverse)
        for position in range(len(items)):
            index, _ = items[position]
            buffer[index] = position
