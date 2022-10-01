from typing import List, Optional

from clean_sample_picking_options import CleanSamplePickingOption, parse_clean_sample_picking_option
from label_noise import LabelNoiseMethod, parse_label_noise_method
from optimizer_options import SchedulerOption, OptimizerOption, parse_optimizer_option, parse_scheduler_option, \
    LossOption, parse_loss_option


class Configuration:
    """
    Contains configuration information for the experiment.
    """

    dataset: str
    lr: float
    lr_adam: float
    use_augmentation: bool
    model: str
    noise_method: LabelNoiseMethod
    noise_level: int
    noise_level_index: int
    trial_index: int
    technique: str
    small_loss_cutoff: int
    cc_loss_coefficient: float
    optimizer_fe: OptimizerOption
    optimizer_if: OptimizerOption
    scheduler: SchedulerOption
    loss: LossOption
    iteration_count: int
    nishi_lr_switch_iteration: int
    desired_samples_per_label: int
    use_unsupervised_asif: bool
    asif_start_epoch: int
    if_training_layer_count: int
    max_if_optimizers_per_iter: int
    if_heads: List[int]
    random_if_heads: int
    overridden_feature_vector_size: Optional[int]
    asif_loss_coefficient: float
    shared_if_head_layer_count: int
    if_reverse_private: bool
    if_reverse_public: bool
    cc_pick_clean_indices: bool
    class_pick_clean_indices: CleanSamplePickingOption
    class_pick_clean_indices_start_epoch: int
    class_pick_clean_indices_start_percentile: float
    class_pick_clean_indices_stop_percentile: float
    cc_handle_positive_cases: bool
    cc_handle_negative_cases: bool
    bad_label_picking_frequency: int
    bad_label_picking_sample_count: int
    bad_label_picking_cycle_count: int
    use_dgr: bool
    gce_q: float
    gce_start_prune_epoch: int
    sgd_decay: float
    sgd_momentum: float
    phuber_tau: float


class Configurator:
    """Loads configurations from a file."""

    @staticmethod
    def load(config_file_path: str) -> List[Configuration]:
        with open(config_file_path, 'r') as in_file:
            configurations = []
            field_names = None
            for line in in_file:
                if line.find('# ') > -1:
                    continue

                if field_names is None:
                    field_names = line.split(',')
                else:
                    field_values = line.split(',')
                    fields = {name.strip(): value.strip() for name, value in zip(field_names, field_values)}

                    configuration = Configuration()
                    configuration.lr = float(fields['LR'])
                    configuration.optimizer_fe = parse_optimizer_option(fields['OptimizerFE'])
                    configuration.dataset = fields['Dataset']

                    if 'TrialIndex' in fields:
                        configuration.trial_index = int(fields['TrialIndex'])
                    else:
                        configuration.trial_index = 0

                    if 'Loss' in fields:
                        configuration.loss = parse_loss_option(fields['Loss'])
                    else:
                        configuration.loss = LossOption.LOSS_CE

                    if 'Model' in fields:
                        configuration.model = fields['Model']
                    else:
                        configuration.model = 'nishi_resnet18'

                    if 'Scheduler' in fields:
                        configuration.scheduler = parse_scheduler_option(fields['Scheduler'])
                    else:
                        configuration.scheduler = SchedulerOption.SCHEDULER_NONE

                    if 'Technique' in fields:
                        configuration.technique = fields['Technique']
                    else:
                        configuration.technique = None

                    if 'OptimizerIF' in fields:
                        configuration.optimizer_if = parse_optimizer_option(fields['OptimizerIF'])
                    else:
                        configuration.optimizer_if = None

                    if 'AdamLR' in fields:
                        configuration.lr_adam = float(fields['AdamLR'])
                    else:
                        configuration.lr_adam = 0.0001

                    if 'IFFETrainLayers' in fields:
                        configuration.if_training_layer_count = int(fields['IFFETrainLayers'])
                    else:
                        configuration.if_training_layer_count = 0

                    if 'UnsupervisedASIF' in fields:
                        configuration.use_unsupervised_asif = fields['UnsupervisedASIF'].upper() == "TRUE"
                    else:
                        configuration.use_unsupervised_asif = False

                    if 'AsifStartEpoch' in fields:
                        configuration.asif_start_epoch = int(fields['AsifStartEpoch'])
                    else:
                        configuration.asif_start_epoch = 0

                    if 'UseAugmentation' in fields:
                        configuration.use_augmentation = fields['UseAugmentation'].upper() == "TRUE"
                    else:
                        configuration.use_augmentation = False

                    if 'NoiseMethod' in fields:
                        configuration.noise_method = parse_label_noise_method(fields['NoiseMethod'])
                    else:
                        configuration.noise_method = LabelNoiseMethod.LABEL_NOISE_NONE

                    if 'NoiseLevel' in fields:
                        configuration.noise_level = int(fields['NoiseLevel'])
                    else:
                        configuration.noise_level = 0

                    if 'NoiseLevelIndex' in fields:
                        configuration.noise_level_index = int(fields['NoiseLevelIndex'])
                    else:
                        configuration.noise_level_index = 0

                    if 'SmallLossCutoff' in fields:
                        configuration.small_loss_cutoff = int(fields['SmallLossCutoff'])
                    else:
                        configuration.small_loss_cutoff = 100

                    if 'CCLossCoefficient' in fields:
                        configuration.cc_loss_coefficient = float(fields['CCLossCoefficient'])
                    else:
                        configuration.cc_loss_coefficient = 1.

                    if 'Iterations' in fields:
                        configuration.iteration_count = int(fields['Iterations'])
                    else:
                        configuration.iteration_count = 0

                    if 'LrSwitchIteration' in fields:
                        configuration.nishi_lr_switch_iteration = int(fields['LrSwitchIteration'])
                    else:
                        configuration.nishi_lr_switch_iteration = configuration.iteration_count + 1

                    if 'SamplesPerLabel' in fields:
                        configuration.desired_samples_per_label = int(fields['SamplesPerLabel'])
                    else:
                        configuration.desired_samples_per_label = 0

                    if 'MaxIfOptimizers' in fields:
                        configuration.max_if_optimizers_per_iter = int(fields['MaxIfOptimizers'])
                    else:
                        configuration.max_if_optimizers_per_iter = 0

                    if 'IFHeads' in fields:
                        if fields['IFHeads'] in ['', '""']:
                            configuration.if_heads = []
                        else:
                            configuration.if_heads = \
                                [int(value) for value in fields['IFHeads'].replace('"', '').split(':')]
                    else:
                        configuration.if_heads = []

                    if 'RandomIFHeads' in fields:
                        if fields['RandomIFHeads'] == '':
                            configuration.random_if_heads = 0
                        else:
                            configuration.random_if_heads = int(fields['RandomIFHeads'])
                    else:
                        configuration.random_if_heads = 0

                    if 'FeatureVectorSize' in fields:
                        if fields['FeatureVectorSize'] == '':
                            configuration.overridden_feature_vector_size = None
                        else:
                            configuration.overridden_feature_vector_size = int(fields['FeatureVectorSize'])
                    else:
                        configuration.overridden_feature_vector_size = None

                    if 'AsifLossCoef' in fields:
                        configuration.asif_loss_coefficient = float(fields['AsifLossCoef'])
                    else:
                        configuration.asif_loss_coefficient = 1.

                    if 'SharedIFLayers' in fields:
                        configuration.shared_if_head_layer_count = int(fields['SharedIFLayers'])
                    else:
                        configuration.shared_if_head_layer_count = 0

                    if 'IFReversePrivate' in fields:
                        configuration.if_reverse_private = fields['IFReversePrivate'].upper() == "TRUE"
                    else:
                        configuration.if_reverse_private = True

                    if 'IFReversePublic' in fields:
                        configuration.if_reverse_public = fields['IFReversePublic'].upper() == "TRUE"
                    else:
                        configuration.if_reverse_public = False

                    if 'CCPickCleans' in fields:
                        configuration.cc_pick_clean_indices = fields['CCPickCleans'].upper() == "TRUE"
                    else:
                        configuration.cc_pick_clean_indices = False

                    if 'ClassPickCleans' in fields:
                        configuration.class_pick_clean_indices = \
                            parse_clean_sample_picking_option(fields['ClassPickCleans'])
                    else:
                        configuration.class_pick_clean_indices = CleanSamplePickingOption.DO_NOT_PICK

                    if 'ClassPickCleansStartEpoch' in fields:
                        configuration.class_pick_clean_indices_start_epoch = \
                            int(fields['ClassPickCleansStartEpoch'])
                    else:
                        configuration.class_pick_clean_indices_start_epoch = 1

                    if 'ClassPickCleansStartPercentile' in fields:
                        configuration.class_pick_clean_indices_start_percentile = \
                            float(fields['ClassPickCleansStartPercentile'])
                    else:
                        configuration.class_pick_clean_indices_start_percentile = 0

                    if 'ClassPickCleansStopPercentile' in fields:
                        configuration.class_pick_clean_indices_stop_percentile = \
                            float(fields['ClassPickCleansStopPercentile'])
                    else:
                        configuration.class_pick_clean_indices_stop_percentile = 100.

                    if 'CCPositive' in fields:
                        configuration.cc_handle_positive_cases = \
                            fields['CCPositive'].upper() == "TRUE"
                    else:
                        configuration.cc_handle_positive_cases = True

                    if 'CCNegative' in fields:
                        configuration.cc_handle_negative_cases = \
                            fields['CCNegative'].upper() == "TRUE"
                    else:
                        configuration.cc_handle_negative_cases = False

                    if 'BadLabelPickingFreq' in fields:
                        configuration.bad_label_picking_frequency = int(fields['BadLabelPickingFreq'])
                    else:
                        configuration.bad_label_picking_frequency = 10

                    if 'BadLabelPickingSampleCount' in fields:
                        configuration.bad_label_picking_sample_count = int(fields['BadLabelPickingSampleCount'])
                    else:
                        configuration.bad_label_picking_sample_count = 1000

                    if 'BadLabelPickingCycleCount' in fields:
                        configuration.bad_label_picking_cycle_count = int(fields['BadLabelPickingCycleCount'])
                    else:
                        configuration.bad_label_picking_cycle_count = 0

                    if 'UseDGR' in fields:
                        configuration.use_dgr = fields['UseDGR'].upper() == "TRUE"
                    else:
                        configuration.use_dgr = True

                    if 'GceQ' in fields:
                        configuration.gce_q = float(fields['GceQ'])
                    else:
                        configuration.gce_q = None

                    if 'GceStartPruneEpoch' in fields:
                        configuration.gce_start_prune_epoch = int(fields['GceStartPruneEpoch'])
                    else:
                        configuration.gce_start_prune_epoch = None

                    if 'SGDDecay' in fields:
                        configuration.sgd_decay = float(fields['SGDDecay'])
                    else:
                        configuration.sgd_decay = 5e-4

                    if 'SGDMomentum' in fields:
                        configuration.sgd_momentum = float(fields['SGDMomentum'])
                    else:
                        configuration.sgd_momentum = 0.9

                    if 'PHuberTau' in fields:
                        configuration.phuber_tau = float(fields['PHuberTau'])
                    else:
                        configuration.phuber_tau = None

                    configurations.append(configuration)
        return configurations


if __name__ == "__main__":
    _configurations = Configurator.load('remote/nishi_baseline_configurations.csv')
    print(_configurations)
