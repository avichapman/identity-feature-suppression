from enum import Enum
from typing import List


class LabelNoiseMethod(Enum):
    """
    Possible types of label noise.
        None               - No noise applied.
        Symmetric          - All labels can be converted to any other with a given probability.
        Asymmetric         - Certain pairs of labels can be flipped with a given probability.
        Instance Dependent - Labels are flipped based on instance difficulty. This is to be computed offline and a file
                             containing labels to flip is to be provided..
        Open Set           - CIFAR100 samples from classes that do not belong in CIFAR10 are randomly included.
    """
    LABEL_NOISE_NONE = 0
    LABEL_NOISE_SYMMETRIC = 1
    LABEL_NOISE_ASYMMETRIC = 2
    LABEL_NOISE_INSTANCE_DEPENDENT = 3
    LABEL_NOISE_OPEN_SET = 4

    def get_short_name(self) -> str:
        """Gets a short description of the technique."""
        if self == LabelNoiseMethod.LABEL_NOISE_NONE:
            return "none"

        if self == LabelNoiseMethod.LABEL_NOISE_SYMMETRIC:
            return "sym"

        if self == LabelNoiseMethod.LABEL_NOISE_ASYMMETRIC:
            return "asym"

        if self == LabelNoiseMethod.LABEL_NOISE_INSTANCE_DEPENDENT:
            return "instance"

        if self == LabelNoiseMethod.LABEL_NOISE_OPEN_SET:
            return "open"

    def get_friendly_name(self) -> str:
        """Gets a human-friendly description of the technique."""
        if self == LabelNoiseMethod.LABEL_NOISE_NONE:
            return "No Label Noise"

        if self == LabelNoiseMethod.LABEL_NOISE_SYMMETRIC:
            return "Symmetric Label Noise"

        if self == LabelNoiseMethod.LABEL_NOISE_ASYMMETRIC:
            return "Asymmetric Label Noise"

        if self == LabelNoiseMethod.LABEL_NOISE_INSTANCE_DEPENDENT:
            return "Instance-Dependent Label Noise"

        if self == LabelNoiseMethod.LABEL_NOISE_OPEN_SET:
            return "Open Set Label Noise"


def get_label_noise_short_options() -> List[str]:
    return ['none', 'sym', 'asym', 'instance', 'open']


def parse_label_noise_method(short_string: str) -> LabelNoiseMethod:
    """
    Converts a short label noise method string into an enumeration.
    :param short_string: The short noise method string to convert.
    :return: An enumeration.
    :raises ValueError if `short_string` is not a valid method string.
    """
    if short_string == 'none':
        return LabelNoiseMethod.LABEL_NOISE_NONE

    if short_string == 'sym':
        return LabelNoiseMethod.LABEL_NOISE_SYMMETRIC

    if short_string == 'asym':
        return LabelNoiseMethod.LABEL_NOISE_ASYMMETRIC

    if short_string == 'instance':
        return LabelNoiseMethod.LABEL_NOISE_INSTANCE_DEPENDENT

    if short_string == 'open':
        return LabelNoiseMethod.LABEL_NOISE_OPEN_SET

    raise ValueError(f"'{short_string}' is not a valid label noise method string.")
