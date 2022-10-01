from enum import Enum
from typing import List


class CleanSamplePickingOption(Enum):
    """
    Possible types of clean sample picking options.
    """

    DO_NOT_PICK = 0
    """
    Do not attempt to work out which samples are clean.
    """

    CLASS_LOSS = 1
    """Train on samples that have previously shown a lower loss in the classification head."""

    IF_HEAD_LOSS = 2
    """Train on samples that have previously shown a lower loss in the IF heads."""

    IF_HEAD_ENTROPY = 3
    """Train on samples that have previously shown a lower logit entropy in the IF heads."""

    IF_HEAD_ENERGY = 4
    """Train on samples that have previously shown a lower logit energy in the IF heads."""

    def get_short_name(self) -> str:
        """Gets a short description of the technique."""
        if self == CleanSamplePickingOption.DO_NOT_PICK:
            return "None"

        if self == CleanSamplePickingOption.CLASS_LOSS:
            return "ClassLoss"

        if self == CleanSamplePickingOption.IF_HEAD_LOSS:
            return "IfHeadLoss"

        if self == CleanSamplePickingOption.IF_HEAD_ENTROPY:
            return "IfHeadEntropy"

        if self == CleanSamplePickingOption.IF_HEAD_ENERGY:
            return "IfHeadEnergy"


def get_clean_sample_picking_options() -> List[str]:
    return ['none', 'classloss', 'ifheadloss', 'ifheadentropy', 'ifheadenergy']


def parse_clean_sample_picking_option(short_string: str) -> CleanSamplePickingOption:
    """
    Converts a short string into an enumeration.
    :param short_string: The short noise method string to convert.
    :return: An enumeration.
    :raises ValueError if `short_string` is not a valid method string.
    """
    if short_string.lower() == 'none':
        return CleanSamplePickingOption.DO_NOT_PICK

    if short_string.lower() == 'classloss':
        return CleanSamplePickingOption.CLASS_LOSS

    if short_string.lower() == 'ifheadloss':
        return CleanSamplePickingOption.IF_HEAD_LOSS

    if short_string.lower() == 'ifheadentropy':
        return CleanSamplePickingOption.IF_HEAD_ENTROPY

    if short_string.lower() == 'ifheadenergy':
        return CleanSamplePickingOption.IF_HEAD_ENERGY

    raise ValueError(f"'{short_string}' is not a valid clean sample picking string.")
