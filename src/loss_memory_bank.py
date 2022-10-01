from datetime import datetime

import torch


class LossMemoryBank:
    """
    Keeps track of the samples with the lowest loss for each training label. This can then be used to find the samples
    with the highest or lowest loss for the purposes of increasing the weight on certain labels or remediating uncertain
    labels.
    """

    def __init__(self, dataset_size: int, use_cuda: bool = None):
        """
        Creates a new loss memory bank.
        :param dataset_size: The total number of samples in the dataset.
        """
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()

        self.dataset_size = dataset_size
        self.sample_measurement_ordering_ascending = torch.tensor([-1] * self.dataset_size)
        self.sample_measurement_ordering_descending = torch.tensor([-1] * self.dataset_size)
        self.sample_metric = torch.tensor([-1.] * self.dataset_size)
        self.sample_labels = torch.tensor([-1] * self.dataset_size)
        self.sample_true_labels = torch.tensor([-1] * self.dataset_size)

        if use_cuda:
            self.sample_measurement_ordering_ascending = self.sample_measurement_ordering_ascending.cuda()
            self.sample_measurement_ordering_descending = self.sample_measurement_ordering_descending.cuda()
            self.sample_metric = self.sample_metric.cuda()
            self.sample_labels = self.sample_labels.cuda()
            self.sample_true_labels = self.sample_true_labels.cuda()

    def add_to_memory(self,
                      sample_indices: torch.Tensor,
                      labels: torch.Tensor,
                      measurement: torch.Tensor,
                      true_labels: torch.Tensor):
        """
        Adds a measurement to memory, along with the apparent truth label.
        :argument sample_indices The indices of the samples in the training batch.
        :argument labels The apparent truth labels for the samples in the training batch.
        :argument measurement The data being measured for each element in the batch, such as loss or entropy.
        :argument true_labels The true labels for collecting measurements vs truth statistics.
        """
        assert sample_indices.size()[0] == labels.size()[0] == len(measurement), \
            "Indices, logits and labels must be the same size."

        self.sample_labels[sample_indices.detach()] = labels.detach()
        self.sample_metric[sample_indices.detach()] = measurement.detach()
        self.sample_true_labels[sample_indices.detach()] = true_labels.detach()

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

    def _build_measurement_cache(self, buffer: torch.Tensor, reverse: bool):
        """
        Once an epoch, this should be called to rebuild the measurement ordering cache.
        :argument buffer: The place to store the indices.
        :argument reverse: If true, build the measurement cache so that the lowest values represent the highest
        measurement.
        """
        items = {}
        for i in range(self.sample_metric.size()[0]):
            _label = self.sample_labels[i].item()
            if _label not in items:
                items[_label] = []
            items[_label].append((i, self.sample_metric[i].item()))

        for _label in items:
            items[_label] = sorted(items[_label], key=lambda datum: datum[1], reverse=reverse)
            for position in range(len(items[_label])):
                index, _ = items[_label][position]
                buffer[index] = position


if __name__ == "__main__":
    bank = LossMemoryBank(dataset_size=50000, use_cuda=True)

    master_sample_indices_ = torch.arange(0, 50000).cuda()
    master_labels_ = torch.randint(0, 2, master_sample_indices_.size()).cuda()
    master_losses_ = torch.rand(master_sample_indices_.size()).cuda()

    start_time = datetime.now()
    for i_ in range(0, 50000, 128):
        sample_indices_ = master_sample_indices_[i_:i_+128]
        labels_ = master_labels_[i_:i_+128]
        losses_ = master_losses_[i_:i_+128]

        bank.add_to_memory(sample_indices_, labels_, losses_, labels_)
    elapsed_time = datetime.now() - start_time
    print("1 - Add to Memory Time:", elapsed_time)

    start_time = datetime.now()
    for i_ in range(0, 50000, 128):
        sample_indices_ = master_sample_indices_[i_:i_+128]
        bank.is_in_bottom_x(sample_indices_, 2)
    elapsed_time = datetime.now() - start_time
    print("1 - Fetch Time:", elapsed_time)

    start_time = datetime.now()
    for i_ in range(0, 50000, 128):
        sample_indices_ = master_sample_indices_[i_:i_+128]
        labels_ = master_labels_[i_:i_+128]
        losses_ = master_losses_[i_:i_+128]

        bank.add_to_memory(sample_indices_, labels_, losses_, labels_)
    elapsed_time = datetime.now() - start_time
    print("2 - Add to Memory Time:", elapsed_time)

    start_time = datetime.now()
    bank.build_lowest_first_measurement_cache()
    elapsed_time = datetime.now() - start_time
    print("2 - Refresh Loss Cache Time:", elapsed_time)

    start_time = datetime.now()
    for i_ in range(0, 50000, 128):
        sample_indices_ = master_sample_indices_[i_:i_+128]
        bank.is_in_bottom_x(sample_indices_, 2)
    elapsed_time = datetime.now() - start_time
    print("2 - Fetch Time:", elapsed_time)
