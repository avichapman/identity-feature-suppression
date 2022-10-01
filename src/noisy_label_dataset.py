from enum import Enum
from typing import List, Union, Set

from dataset_cifar10 import DatasetCIFAR10
from dataset_cifar100 import DatasetCIFAR100
from dataset_fashion_mnist import DatasetFashionMNIST
from dataset_fgvc_aircraft import DatasetFgvcAircraft
from dataset_mnist import DatasetMNIST
from label_noise import LabelNoiseMethod


class NoisyDatasetType(Enum):
    """
    Possible types of noisy dataset.
    """
    NOISY_DATASET_CIFAR10 = 0
    NOISY_DATASET_CIFAR100 = 1
    NOISY_DATASET_MNIST = 2
    NOISY_DATASET_FGVC_AIRCRAFT = 3
    NOISY_DATASET_FASHION_MNIST = 4

    def get_short_name(self) -> str:
        """Gets a short description of the technique."""
        if self == NoisyDatasetType.NOISY_DATASET_CIFAR10:
            return "cifar10"

        if self == NoisyDatasetType.NOISY_DATASET_CIFAR100:
            return "cifar100"

        if self == NoisyDatasetType.NOISY_DATASET_MNIST:
            return "mnist"

        if self == NoisyDatasetType.NOISY_DATASET_FASHION_MNIST:
            return "fashion_mnist"

        if self == NoisyDatasetType.NOISY_DATASET_FGVC_AIRCRAFT:
            return "fgvc"


def get_label_noise_short_options() -> List[str]:
    return ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'fgvc']


def parse_noisy_dataset_type(short_string: str) -> NoisyDatasetType:
    """
    Converts a short label noise method string into an enumeration.
    :param short_string: The short noise method string to convert.
    :return: An enumeration.
    :raises ValueError if `short_string` is not a valid method string.
    """
    if short_string.lower() == 'cifar10':
        return NoisyDatasetType.NOISY_DATASET_CIFAR10

    if short_string.lower() == 'cifar100':
        return NoisyDatasetType.NOISY_DATASET_CIFAR100

    if short_string.lower() == 'mnist':
        return NoisyDatasetType.NOISY_DATASET_MNIST

    if short_string.lower() == 'fashion_mnist':
        return NoisyDatasetType.NOISY_DATASET_FASHION_MNIST

    if short_string.lower() == 'fgvc':
        return NoisyDatasetType.NOISY_DATASET_FGVC_AIRCRAFT

    raise ValueError(f"'{short_string}' is not a valid noisy dataset type string.")


NoisyDataset = Union[DatasetCIFAR10, DatasetCIFAR100, DatasetFgvcAircraft, DatasetMNIST, DatasetFashionMNIST]


def create_noisy_dataset(dataset_type: NoisyDatasetType,
                         data_dir: str,
                         train: bool,
                         randomize: bool = None,
                         apply_data_augmentation: bool = False,
                         label_file_dir: str = None,
                         noise_method: LabelNoiseMethod = None,
                         noise_level: int = None,
                         trial_index: int = None,
                         use_single_group: bool = False,
                         return_unlabelled_samples: bool = False,
                         samples_per_label: int = 0,
                         gpu_count: int = None,
                         batch_size: int = None,
                         label_remediation_file: str = None,
                         probably_false_sample_indices: Set[int] = None) -> NoisyDataset:
    """
    Creates a new noisy dataset of the type specified by `dataset_type`.

    :argument data_dir The path to the dataset data.
    :argument dataset_type The type of dataset to create.
    :argument train If true, use training data.
    :argument randomize If provided and true, randomise the data. If not provided, defaults to the value of `train`.
    :argument apply_data_augmentation If true, apply data augmentation techniques to reduce overfitting.
    :argument label_file_dir The directory where the label noise files live.
    :argument noise_method The method to use to generate the label noise. See the enum doc for details.
    :argument noise_level The chance of changing a label.
    :argument trial_index Zero-based index of version of the noise data to load.
    :argument use_single_group If true, have only one group for ASIF.
    :argument return_unlabelled_samples If true, return all samples in dataset, but only apply labels to some of them.
    :argument samples_per_label Number of samples per label to train on. Zero means all.
    :argument gpu_count: If provided, return batches such that the right samples go to the right GPUs. This can be
    used to ensure that the first 50 classes are handled on one GPU while the next 50 are handled on another GPU.
    If provided, `batch_size` must also be provided.
    :argument batch_size The expected batch size to be trained on. This must be provided if gpu_count is provided.
    :argument label_remediation_file If provided, points to a file containing whether labels are suspected to be
        incorrect.
    :argument probably_false_sample_indices If provided, sets certain sample indices to be assumed to have false labels.
    """
    if dataset_type == NoisyDatasetType.NOISY_DATASET_CIFAR10:
        return DatasetCIFAR10(train=train,
                              data_dir=data_dir,
                              randomize=randomize,
                              apply_data_augmentation=apply_data_augmentation,
                              label_file_dir=label_file_dir,
                              noise_method=noise_method,
                              noise_level=noise_level,
                              trial_index=trial_index,
                              use_single_group=use_single_group,
                              return_unlabelled_samples=return_unlabelled_samples,
                              desired_samples_per_label=samples_per_label,
                              gpu_count=gpu_count,
                              batch_size=batch_size,
                              label_remediation_file=label_remediation_file,
                              probably_false_sample_indices=probably_false_sample_indices)
    elif dataset_type == NoisyDatasetType.NOISY_DATASET_CIFAR100:
        return DatasetCIFAR100(train=train,
                               data_dir=data_dir,
                               randomize=randomize,
                               apply_data_augmentation=apply_data_augmentation,
                               label_file_dir=label_file_dir,
                               noise_method=noise_method,
                               noise_level=noise_level,
                               trial_index=trial_index,
                               use_single_group=use_single_group,
                               return_unlabelled_samples=return_unlabelled_samples,
                               desired_samples_per_label=samples_per_label,
                               gpu_count=gpu_count,
                               batch_size=batch_size,
                               label_remediation_file=label_remediation_file,
                               probably_false_sample_indices=probably_false_sample_indices)
    elif dataset_type == NoisyDatasetType.NOISY_DATASET_MNIST:
        return DatasetMNIST(train=train,
                            data_dir=data_dir,
                            randomize=randomize,
                            apply_data_augmentation=apply_data_augmentation,
                            label_file_dir=label_file_dir,
                            noise_method=noise_method,
                            noise_level=noise_level,
                            trial_index=trial_index,
                            use_single_group=use_single_group,
                            return_unlabelled_samples=return_unlabelled_samples,
                            desired_samples_per_label=samples_per_label,
                            gpu_count=gpu_count,
                            batch_size=batch_size,
                            label_remediation_file=label_remediation_file,
                            probably_false_sample_indices=probably_false_sample_indices)
    elif dataset_type == NoisyDatasetType.NOISY_DATASET_FASHION_MNIST:
        return DatasetFashionMNIST(train=train,
                                   data_dir=data_dir,
                                   randomize=randomize,
                                   apply_data_augmentation=apply_data_augmentation,
                                   label_file_dir=label_file_dir,
                                   noise_method=noise_method,
                                   noise_level=noise_level,
                                   trial_index=trial_index,
                                   use_single_group=use_single_group,
                                   return_unlabelled_samples=return_unlabelled_samples,
                                   desired_samples_per_label=samples_per_label,
                                   gpu_count=gpu_count,
                                   batch_size=batch_size,
                                   label_remediation_file=label_remediation_file,
                                   probably_false_sample_indices=probably_false_sample_indices)
    elif dataset_type == NoisyDatasetType.NOISY_DATASET_FGVC_AIRCRAFT:
        return DatasetFgvcAircraft(data_dir=data_dir,
                                   is_train=train,
                                   is_eval=not train,
                                   is_test=False,
                                   randomize=randomize,
                                   apply_data_augmentation=apply_data_augmentation,
                                   desired_samples_per_label=samples_per_label,
                                   gpu_count=gpu_count,
                                   batch_size=batch_size)
    else:
        raise ValueError(f"'{dataset_type}' is not a supported noisy dataset.")
