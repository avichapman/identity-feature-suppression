import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Set

from label_noise import LabelNoiseMethod


class DatasetCIFAR100(Dataset):
    """
    Wrapper around the standard CIFAR100 dataset that allows us to introduce label noise.

    For symmetric, asymmetric and instance dependent noise, a label noise file is loaded and used.

    There are multiple versions of each label noise file. This allows for multiple trials with different noise each
    time.
    """

    def __init__(self,
                 train: bool,
                 data_dir: str,
                 randomize: bool = None,
                 apply_data_augmentation: bool = False,
                 label_file_dir: str = None,
                 noise_method: LabelNoiseMethod = None,
                 noise_level: int = None,
                 trial_index: int = None,
                 desired_samples_per_label: int = 0,
                 use_single_group: bool = False,
                 return_unlabelled_samples: bool = False,
                 gpu_count: int = None,
                 batch_size: int = None,
                 label_remediation_file: str = None,
                 probably_false_sample_indices: Set[int] = None):
        """
        Creates a new dataset.

        :argument train If true, use training data.
        :argument data_dir The path to the dataset data.
        :argument randomize If provided and true, randomise the data. If not provided, defaults to the value of `train`.
        :argument apply_data_augmentation If true, apply data augmentation techniques to reduce overfitting.
        :argument label_file_dir The directory where the label noise files live.
        :argument noise_method The method to use to generate the label noise. See the enum doc for details.
        :argument noise_level The chance of changing a label.
        :argument trial_index Zero-based index of version of the noise data to load.
        :argument desired_samples_per_label If positive, only return this number of samples of each label during
        training. If zero, use all of the data.
        :argument use_single_group If true, have only one group for ASIF.
        :argument return_unlabelled_samples If true, return all samples in dataset, but only apply labels to some of
        them.
        :argument gpu_count: If provided, return batches such that the right samples go to the right GPUs. This can be
        used to ensure that the first 50 classes are handled on one GPU while the next 50 are handled on another GPU.
        If provided, `batch_size` must also be provided.
        :argument batch_size The expected batch size to be trained on. This must be provided if gpu_count is provided.
        :argument label_remediation_file If provided, points to a file containing whether labels are suspected to be
        incorrect.
        :argument probably_false_sample_indices If provided, sets certain sample indices to be assumed to have false
        labels.
        """
        if randomize is None:
            randomize = train

        if train:
            assert label_file_dir is not None, "`label_file_dir` must be provided in training mode."
            assert noise_method is not None, "`noise_method` must be provided in training mode."
            assert noise_level is not None, "`noise_level` must be provided in training mode."
            assert trial_index is not None, "`trial_index` must be provided in training mode."
            assert 0 <= noise_level <= 100, "Noise Level must be in range [0, 100]."

        if gpu_count is not None:
            assert batch_size is not None and batch_size > 0, "`batch_size` must be set if `gpu_count` is set."

        if noise_method is None:
            noise_method = LabelNoiseMethod.LABEL_NOISE_NONE

        self.train = train
        self.randomize = randomize
        self.apply_data_augmentation = apply_data_augmentation
        self.transform = self._get_transform(augmented=False)
        self.transform_augmented = self._get_transform(augmented=True)
        self.noise_level = noise_level
        self.noise_method = noise_method
        self.trial_index = trial_index
        self.desired_samples_per_label = desired_samples_per_label
        self.use_single_group = use_single_group
        self.return_unlabelled_samples = return_unlabelled_samples
        self.gpu_count = gpu_count
        self.batch_size = batch_size

        if label_file_dir is not None:
            self.labels_dir = os.path.join(label_file_dir, "cifar100")
        else:
            self.labels_dir = None

        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=train,
                                                download=True, transform=self.transform)
        self.class_names = dataset.classes
        self.class_count = len(self.class_names)
        self.feature_count = 3

        corrupted_labels = self._add_label_noise(dataset)

        if label_remediation_file is not None:
            suspected_false_labels = self._load_label_remediation_file(label_remediation_file)
        else:
            suspected_false_labels = None

        if probably_false_sample_indices is not None:
            injected_false_labels = probably_false_sample_indices
        else:
            injected_false_labels = set()

        self.original_length = len(dataset.data)
        if self.apply_data_augmentation:
            # We return both augmented and un-augmented samples...
            self.original_length = self.original_length * 2

        self.data, self.samples_per_group, self.samples_per_label = \
            self._sort_data_into_groups(dataset.data,
                                        dataset.targets,
                                        corrupted_labels,
                                        suspected_false_labels,
                                        injected_false_labels)

        # Allocate data to individual GPUs...
        if self.gpu_count is not None:
            data = [[] for _ in range(self.gpu_count)]
            classes_per_gpu = self.class_count // self.gpu_count
            for sample, label_id, true_label_id, group_id, group_membership_id, probably_incorrect_label in self.data:
                gpu_index = label_id // classes_per_gpu
                data[gpu_index].append((sample,
                                        label_id,
                                        true_label_id,
                                        group_id,
                                        group_membership_id,
                                        probably_incorrect_label))
            self.data = data
        else:
            # Only one GPU, all data goes in it...
            self.data = [self.data]

    @staticmethod
    def _get_transform(augmented: bool) -> transforms.Compose:
        """
        Returns the tensor transform to apply to the dataset.
        """
        if augmented:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def _sort_data_into_groups(self,
                               samples: np.ndarray,
                               labels: List[int],
                               corrupted_labels: List[int],
                               suspected_false_labels: Optional[List[Tuple[int, int, bool]]],
                               injected_false_labels: Optional[Set[int]]) -> \
            Tuple[List[Tuple[np.ndarray, int, int, int, int, int]], Dict[int, int], Dict[int, int]]:
        """
        Sorts data into groups and provides each sample with an ID that is unique within its apparent label
        type.

        Note: Due to label noise, the group id may not be unique within each samples actual label type.

        :param samples: The sample data.
        :param labels: The true labels for each sample in `samples`.
        :param corrupted_labels: The noisy labels for each sample in `samples`.
        :param suspected_false_labels: For each sample, whether or not we suspect its label may be incorrect.
        :param injected_false_labels: Manually set list of sample indices suspected to have false labels.
        :return: A tuple containing:
        - A list of tuples with the sample, apparent label, true label, group id, unique group membership id and whether
          we suspect the label of being incorrect.
        - A dictionary of sample counts for each group
        - A dictionary of sample counts for each label
        """
        _data = []  # type: List[Tuple[np.ndarray, int, int, int, int]]
        _samples_per_group = {}  # type: Dict[int, int]
        _samples_per_label = {}  # type: Dict[int, int]
        for i in range(len(samples)):
            original_label_id = labels[i]
            label_id = corrupted_labels[i]

            if suspected_false_labels is not None:
                sample_id, suspicious_label_id, is_probably_falsely_labelled = suspected_false_labels[i]
                if is_probably_falsely_labelled:
                    assert suspicious_label_id == label_id, "The label remediation is misaligned."
            else:
                is_probably_falsely_labelled = False

            if i in injected_false_labels:
                is_probably_falsely_labelled = True

            if not is_probably_falsely_labelled:
                if self.use_single_group:
                    group_id = 0
                else:
                    group_id = label_id

                if label_id not in _samples_per_label:
                    _samples_per_label[label_id] = 0
                if group_id not in _samples_per_group:
                    _samples_per_group[group_id] = 0
                group_member_id = _samples_per_group[group_id]  # Assign a group ID unique to each label type
            else:
                group_id = -1
                group_member_id = -1

            if self.desired_samples_per_label == 0 or \
                    _samples_per_label[label_id] < self.desired_samples_per_label or \
                    self.return_unlabelled_samples or is_probably_falsely_labelled:

                # If we are returning unlabelled samples and we are past our quota, return -1...
                if is_probably_falsely_labelled:
                    # Return the label. The trainer can decide what to do with it...
                    _label_to_return = label_id
                elif self.desired_samples_per_label == 0:
                    # Return all labels...
                    _label_to_return = label_id
                elif _samples_per_label[label_id] >= self.desired_samples_per_label and self.return_unlabelled_samples:
                    _label_to_return = -1
                else:
                    _label_to_return = label_id

                # noinspection PyUnresolvedReferences
                _data.append((samples[i],
                              _label_to_return,
                              original_label_id,
                              group_id,
                              group_member_id,
                              1 if is_probably_falsely_labelled else 0))
                if group_id >= 0:
                    _samples_per_group[group_id] += 1

                    if _label_to_return > -1:
                        _samples_per_label[label_id] += 1

        return _data, _samples_per_group, _samples_per_label

    @staticmethod
    def _load_label_remediation_file(path: str) -> List[Tuple[int, int, bool]]:
        """
        Loads a CSV containing a list of booleans. Each boolean determines whether that sample's label is suspected
        to be incorrect.
        :param path: The path to the file to open.
        :return: A list of booleans.
        """
        data = []
        with open(path, 'r') as infile:
            for line in infile:
                parts = line.split(',')
                sample_id = int(parts[0])
                apparent_label = int(parts[1])
                is_probably_false = parts[2].strip() == 'True'
                data.append((sample_id, apparent_label, is_probably_false))
        return data

    def _add_label_noise(self, data: torchvision.datasets.CIFAR100) -> List[int]:
        """
        Applies noise using the method selected in 'self.noise_method.

        :param data: A CIFAR100 dataset.
        :return: The corrupted labels for each sample in the dataset.
        """
        if self.noise_method == LabelNoiseMethod.LABEL_NOISE_SYMMETRIC:
            label_file = os.path.join(self.labels_dir, f"sym.{self.noise_level}.{self.trial_index}.csv")
            return self._add_label_noise_from_file(data, label_file)
        elif self.noise_method == LabelNoiseMethod.LABEL_NOISE_ASYMMETRIC:
            label_file = os.path.join(self.labels_dir, f"asym.{self.noise_level}.{self.trial_index}.csv")
            return self._add_label_noise_from_file(data, label_file)
        elif self.noise_method == LabelNoiseMethod.LABEL_NOISE_INSTANCE_DEPENDENT:
            raise ValueError('Instance Noise is not supported with the CIFAR100 dataset.')
        elif self.noise_method == LabelNoiseMethod.LABEL_NOISE_OPEN_SET:
            raise ValueError('Open Set Noise is not supported with the CIFAR100 dataset.')
        else:
            return self._add_null_label_noise(data)

    @staticmethod
    def _add_label_noise_from_file(data: torchvision.datasets.CIFAR100, file_path: str) -> List[int]:
        """Applies pre-computed label noise from a CSV file.

        :param data: A CIFAR100 dataset.
        :param file_path: The path to the label noise file.
        :return: The corrupted labels for each sample in the dataset.
        """

        labels = []
        current_sample = -1
        with open(file_path, 'r') as infile:
            for line in infile:
                if line.startswith('#'):
                    # Skip comments...
                    continue

                if current_sample >= 0:
                    parts = line.split(",")
                    if data.targets[current_sample] != int(parts[0]):
                        raise ValueError("Provided label file is misaligned. True label doesn't match dataset.")

                    labels.append(int(parts[1]))
                current_sample += 1

        return labels

    @staticmethod
    def _add_null_label_noise(data: torchvision.datasets.CIFAR100) -> List[int]:
        """
        Null case where no label noise is applied.

        :param data: A CIFAR100 dataset.
        :return: The 'corrupted' labels for each sample in the dataset. In this case, no change has been made.
        """
        return data.targets

    def mark_as_probably_false_label(self, index: int):
        """
        If the trainer determines that a sample with a given `index`'s label is probably false, it can call this method.
        In future iterations, the `is_probably_incorrect_label` value will be '1'.
        :param index: The index of the sample in question.
        """
        assert index < len(self), "Index should never get more than the max allowed."

        sample, label_id, true_label_id, group_id, group_membership_id, _ = self.data[0][index]
        self.data[0][index] = sample, label_id, true_label_id, group_id, group_membership_id, 1

    def is_probably_false_label(self, index: int) -> bool:
        """
        If the trainer determines that a sample with a given `index`'s label is probably false, this will return 'True'.
        :param index: The index of the sample in question.
        """
        assert index < len(self), "Index should never get more than the max allowed."

        _, _, _, _, _, is_probably_false = self.data[0][index]
        return is_probably_false == 1

    def __len__(self):
        return self.original_length

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int, int, int, int]:
        """Retrieves a sample and its associated label and group.

        Returns:
            sample - An image tensor.
            label_id - The index of the sample's apparent label. Can be -1 if no label is provided.
            true_label_id - The index of the sample's true label. Do not train on this.
            group_id - An index for the group within which this sample's membership id is unique.
            group_membership_id - An index for the sample that is unique amongst its peers with the same apparent label.
            sample_index - The unique index of the sample. Used for tracking individual sample stats
            is_probably_incorrect_label - 1 if we suspect the label may be false.
        """
        assert index < len(self), "Index should never get more than the max allowed."

        # If we are applying augmentation, the dataset is twice as long. We return each sample once augmented and once
        # without augmentation...
        if self.apply_data_augmentation:
            if index % 2 == 0:
                transform = self.transform
                use_augmentation = False
            else:
                transform = self.transform_augmented
                use_augmentation = True
            index = index // 2
        else:
            transform = self.transform
            use_augmentation = False

        # Work out GPU to train this one. If there are two GPUs and this is the first half of a batch, return data for
        # the first GPU...
        if self.gpu_count is not None:
            index_within_batch = index % self.batch_size
            batch_positions_per_gpu = self.batch_size // self.gpu_count
            gpu_index = index_within_batch // batch_positions_per_gpu

            index_within_data = random.randint(0, len(self.data[gpu_index]) - 1)
            sample_index = index  # TODO: Fix this to make the returned sample deterministic
        else:
            # We may want to train on a subset of the original data, roll around to the start when we reach the end of
            # the training data. This means that each epoch will have the same number of iterations.
            gpu_index = 0
            if self.randomize:
                index_within_data = random.randint(0, len(self.data[gpu_index]) - 1)
            else:
                index_within_data = index
            sample_index = index_within_data

        sample, label_id, true_label_id, group_id, group_membership_id, is_probably_incorrect_label = \
            self.data[gpu_index][index_within_data]

        # Load it as an image and then apply any transforms...
        sample = Image.fromarray(sample)
        sample = transform(sample)

        # If we are augmenting, we do not perform ASIF...
        if use_augmentation:
            group_id = -1
            group_membership_id = -1

        return sample, label_id, true_label_id, group_id, group_membership_id, sample_index, is_probably_incorrect_label


if __name__ == "__main__":
    _batch_size = 128
    _dataset = DatasetCIFAR100(train=True,
                               randomize=False,
                               data_dir="./data",
                               label_file_dir="../../label_noise",
                               noise_method=LabelNoiseMethod.LABEL_NOISE_NONE,
                               noise_level=20,
                               apply_data_augmentation=True,
                               use_single_group=False,
                               return_unlabelled_samples=False,
                               desired_samples_per_label=0,
                               trial_index=1)
    _dataset.mark_as_probably_false_label(100)
    samples_per_group_membership = {}
    samples_per_group = {}
    samples_per_label = {}
    samples_per_true_label = {}
    count = 0
    wrong_label_count = 0
    wrong_labels_found_out_count = 0
    flagged_labels_count = 0

    total_false_positive = 0
    total_false_negative = 0
    total_true_positive = 0
    total_true_negative = 0

    labels_per_batch_index = {}

    for _sample, _label_id, _true_label_id, _group_id, _group_membership_id, _sample_index, _probably_wrong in _dataset:
        if _probably_wrong == 1:
            print(count)

        _batch_index = count % _batch_size
        if _batch_index not in labels_per_batch_index:
            labels_per_batch_index[_batch_index] = []
        labels_per_batch_index[_batch_index].append(_label_id)

        if _label_id not in samples_per_label:
            samples_per_label[_label_id] = 0
        samples_per_label[_label_id] += 1

        if _group_id not in samples_per_group:
            samples_per_group[_group_id] = 0
        samples_per_group[_group_id] += 1

        if _true_label_id not in samples_per_true_label:
            samples_per_true_label[_true_label_id] = 0
        samples_per_true_label[_true_label_id] += 1

        if _group_membership_id not in samples_per_group_membership:
            samples_per_group_membership[_group_membership_id] = 0
        samples_per_group_membership[_group_membership_id] += 1
        count += 1

        if _label_id != _true_label_id:
            wrong_label_count += 1

        if _label_id != _true_label_id and _probably_wrong == 1:
            wrong_labels_found_out_count += 1

        if _probably_wrong == 1:
            flagged_labels_count += 1

        if _label_id != _true_label_id and _probably_wrong == 1:
            total_true_positive += 1
        if _label_id != _true_label_id and _probably_wrong == 0:
            total_false_negative += 1
        if _label_id == _true_label_id and _probably_wrong == 1:
            total_false_positive += 1
        if _label_id == _true_label_id and _probably_wrong == 0:
            total_true_negative += 1

        if count >= len(_dataset):
            break

    print('Class Count:', _dataset.class_count)
    print('Class Names:', _dataset.class_names)
    print('Labels Wrong (%):', wrong_label_count * 100. / count)
    print('Wrong Labels Flagged (%):', wrong_labels_found_out_count * 100. / count)
    print('Flagged Labels (%):', flagged_labels_count * 100. / count)
    print('Samples per Apparent Label:', samples_per_label)
    print('Samples per True Label:', samples_per_true_label)
    print('Samples per Group ID:', samples_per_group)
    print('Samples per Group Membership Id:', samples_per_group_membership)
