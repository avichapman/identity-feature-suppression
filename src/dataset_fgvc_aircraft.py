import os
import random
from typing import List, Tuple, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DatasetFgvcAircraft(Dataset):
    """
    Provides FGVC Aircraft image data for training on variant, manufacturer or family information.

    Which information is used depends on the value of `self.variable`.

    @techreport{maji13fine-grained,
       title         = {Fine-Grained Visual Classification of Aircraft},
       author        = {S. Maji and J. Kannala and E. Rahtu
                        and M. Blaschko and A. Vedaldi},
       year          = {2013},
       archivePrefix = {arXiv},
       eprint        = {1306.5151},
       primaryClass  = "cs-cv",
    }
    """

    def __init__(self,
                 data_dir: str,
                 is_train: bool = True,
                 is_eval: bool = False,
                 is_test: bool = False,
                 randomize: bool = None,
                 apply_data_augmentation: bool = False,
                 desired_samples_per_label: int = 0,
                 gpu_count: int = None,
                 batch_size: int = None):
        """
        Creates a new dataset.

        Only one of the following can be set:
        :argument is_train If true, use training data.
        :argument is_eval If true, use eval data.
        :argument is_test If true, use testing data.

        :argument data_dir The path to the dataset data.
        :argument randomize If provided and true, randomise the data. If not provided, defaults to the value of `train`.
        :argument apply_data_augmentation If true, apply data augmentation techniques to reduce overfitting.
        :argument desired_samples_per_label If positive, only return this number of samples of each label during
        training. If zero, use all of the data
        :argument gpu_count: If provided, return batches such that the right samples go to the right GPUs. This can be
        used to ensure that the first 50 classes are handled on one GPU while the next 50 are handled on another GPU.
        If provided, `batch_size` must also be provided.
        :argument batch_size The expected batch size to be trained on. This must be provided if gpu_count is provided.
        """
        assert (is_train and not is_eval and not is_test) or \
            (not is_train and is_eval and not is_test) or \
            (not is_train and not is_eval and is_test), "Only one of 'train', 'eval' or 'test' can be set."

        if randomize is None:
            randomize = is_train

        if gpu_count is not None:
            assert batch_size is not None and batch_size > 0, "`batch_size` must be set if `gpu_count` is set."

        self.is_train = is_train
        self.is_eval = is_eval
        self.is_test = is_test
        self.randomize = randomize
        self.apply_data_augmentation = apply_data_augmentation
        self.transform = self._get_transform()
        self.desired_samples_per_label = desired_samples_per_label
        self.gpu_count = gpu_count
        self.batch_size = batch_size
        self.variable = 'variant'

        # Load the data...
        self.feature_count = 3
        self.class_names, self.class_indices = self._load_classes(data_dir)
        self.class_count = len(self.class_names)
        data = self._load_data(data_dir)
        self.original_length = len(data)
        self.data, self.samples_per_group, self.samples_per_label = self._sort_data_into_groups(data)

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

    def _sort_data_into_groups(self, data: List[Tuple[str, int]]) -> \
            Tuple[List[Tuple[str, int, int, int, int, int]], Dict[int, int], Dict[int, int]]:
        """
        Sorts data into groups and provides each sample with an ID that is unique within its apparent label
        type.

        :param data: The sample data and labels.
        :return: A tuple containing:
        - A list of tuples with the sample, apparent label, true label, group id, unique group membership id and whether
          we suspect the label of being incorrect.
        - A dictionary of sample counts for each group
        - A dictionary of sample counts for each label
        """
        _data = []  # type: List[Tuple[str, int, int, int, int]]
        _samples_per_group = {}  # type: Dict[int, int]
        _samples_per_label = {}  # type: Dict[int, int]
        for i in range(len(data)):
            image_file, original_label_id = data[i]
            label_id = original_label_id

            is_probably_falsely_labelled = False

            group_id = label_id

            if label_id not in _samples_per_label:
                _samples_per_label[label_id] = 0
            if group_id not in _samples_per_group:
                _samples_per_group[group_id] = 0
            group_member_id = _samples_per_group[group_id]  # Assign a group ID unique to each label type

            if self.desired_samples_per_label == 0 or \
                    _samples_per_label[label_id] < self.desired_samples_per_label or is_probably_falsely_labelled:

                _label_to_return = label_id

                # noinspection PyUnresolvedReferences
                _data.append((image_file,
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

    def _load_classes(self, data_dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Loads all classes in the dataset and returns a list of class names as well as a dictionary to look
        up indices from names.
        """
        classes = []
        class_indices = {}
        with open(os.path.join(data_dir, f'{self.variable}s.txt'), 'r') as infile:
            for line in infile:
                class_indices[line.strip()] = len(classes)
                classes.append(line.strip())
        return classes, class_indices

    def _load_data(self, data_dir: str) -> List[Tuple[str, int]]:
        """
        Loads all variant classes in the dataset and returns a list of class names.
        """
        if self.is_train:
            data_file_path = os.path.join(data_dir, f'images_{self.variable}_train.txt')
        elif self.is_eval:
            data_file_path = os.path.join(data_dir, f'images_{self.variable}_val.txt')
        else:
            data_file_path = os.path.join(data_dir, f'images_{self.variable}_test.txt')

        data = []
        with open(data_file_path, 'r') as infile:
            for line in infile:
                parts = line.split(' ')
                image_name = os.path.join(data_dir, 'images', parts[0].strip() + '.jpg')

                class_name = line[len(parts[0].strip()):].strip()
                image_class_index = self.class_indices[class_name]
                data.append((image_name, image_class_index))

        return data

    def _get_transform(self) -> transforms.Compose:
        """
        Returns the tensor transform to apply to the dataset.
        """
        if self.is_train and self.apply_data_augmentation:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return self.original_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int, int, int, int]:
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

        img_path, label_id, true_label_id, group_id, group_membership_id, is_probably_incorrect_label = \
            self.data[gpu_index][index_within_data]

        img = Image.open(img_path).convert('RGB')
        img = img.resize((400, 300), resample=Image.BICUBIC)
        img = self.transform(img)

        return img, label_id, true_label_id, group_id, group_membership_id, sample_index, is_probably_incorrect_label
