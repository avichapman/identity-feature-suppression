from enum import Enum
from typing import List


class OptimizerOption(Enum):
    """
    Possible types of optimizer training options.
    """
    OPTIMIZER_SGD = 0
    OPTIMIZER_ADAM = 1

    def get_short_name(self) -> str:
        """Gets a short description of the technique."""
        if self == OptimizerOption.OPTIMIZER_SGD:
            return "SGD"

        if self == OptimizerOption.OPTIMIZER_ADAM:
            return "Adam"


def get_optimizer_options() -> List[str]:
    return ['sgd', 'adam']


def parse_optimizer_option(short_string: str) -> OptimizerOption:
    """
    Converts a short optimizer string into an enumeration.
    :param short_string: The short noise method string to convert.
    :return: An enumeration.
    :raises ValueError if `short_string` is not a valid method string.
    """
    if short_string.lower() == 'sgd':
        return OptimizerOption.OPTIMIZER_SGD

    if short_string.lower() == 'adam':
        return OptimizerOption.OPTIMIZER_ADAM

    raise ValueError(f"'{short_string}' is not a valid optimizer string.")


class SchedulerOption(Enum):
    """
    Possible types of training scheduling options.
        None               - No adjustments made to the LR.
        Common Learning    - LR reduced 10% at 19.5k, 25k and 30k iterations
        IDN                - Like in IDN paper, recreate SGD anew with each epoch.
                             LR is calculated the way Chen et al did.
        Nishi              - Drop to 10% after a certain epoch.
        GCE                - LR divided by 10 after 40 and 80 epochs
    """
    SCHEDULER_NONE = 0
    SCHEDULER_COMMON_LEARNING = 1
    SCHEDULER_IDN = 2
    SCHEDULER_NISHI = 3
    SCHEDULER_GCE = 4
    SCHEDULER_PHUBER = 5

    def get_short_name(self) -> str:
        """Gets a short description of the technique."""
        if self == SchedulerOption.SCHEDULER_NONE:
            return "None"

        if self == SchedulerOption.SCHEDULER_COMMON_LEARNING:
            return "Common"

        if self == SchedulerOption.SCHEDULER_IDN:
            return "IDN"

        if self == SchedulerOption.SCHEDULER_NISHI:
            return "Nishi"

        if self == SchedulerOption.SCHEDULER_GCE:
            return "GCE"

        if self == SchedulerOption.SCHEDULER_PHUBER:
            return "PHuber"


def get_scheduler_options() -> List[str]:
    return ['none', 'common', 'idn', 'nishi', 'gce', 'phuber']


def parse_scheduler_option(short_string: str) -> SchedulerOption:
    """
    Converts a short scheduler string into an enumeration.
    :param short_string: The short loss method string to convert.
    :return: An enumeration.
    :raises ValueError if `short_string` is not a valid method string.
    """
    if short_string.lower() == 'none':
        return SchedulerOption.SCHEDULER_NONE

    if short_string.lower() == 'commonlearning':
        return SchedulerOption.SCHEDULER_COMMON_LEARNING

    if short_string.lower() == 'idn':
        return SchedulerOption.SCHEDULER_IDN

    if short_string.lower() == 'nishi':
        return SchedulerOption.SCHEDULER_NISHI

    if short_string.lower() == 'gce':
        return SchedulerOption.SCHEDULER_GCE

    if short_string.lower() == 'phuber':
        return SchedulerOption.SCHEDULER_PHUBER

    raise ValueError(f"'{short_string}' is not a valid scheduler string.")


class LossOption(Enum):
    """
    Possible types of training loss for the feature classification layer.
        CE  - Cross Entropy Loss
        NLL - Negative Log Likelihood Loss
    """
    LOSS_CE = 0
    LOSS_NLL = 1

    def get_short_name(self) -> str:
        """Gets a short description of the technique."""
        if self == LossOption.LOSS_CE:
            return "CrossEntropy"

        if self == LossOption.LOSS_NLL:
            return "NLL"


def get_loss_options() -> List[str]:
    return ['ce', 'nll']


def parse_loss_option(short_string: str) -> LossOption:
    """
    Converts a short loss string into an enumeration.
    :param short_string: The short loss method string to convert.
    :return: An enumeration.
    :raises ValueError if `short_string` is not a valid method string.
    """
    if short_string.lower() == 'ce':
        return LossOption.LOSS_CE

    if short_string.lower() == 'nll':
        return LossOption.LOSS_NLL

    raise ValueError(f"'{short_string}' is not a valid loss string.")
