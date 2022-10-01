import argparse
import gc
import math
import os
from datetime import datetime
from itertools import chain

from torch.cuda import device

from accuracy import Accuracy
from average_meter import AverageMeter

import matplotlib.pyplot as plt
import numpy as np

import statistics

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils.data
from typing import List, Dict, Union, Tuple, Iterator, Optional

from clean_sample_picking_options import CleanSamplePickingOption
from configuration import Configurator
from constrastive_clusterer import ConstrastiveClusterer
from generalized_cross_entropy_loss import GCELoss
from label_noise import LabelNoiseMethod
from loss_memory_bank import LossMemoryBank
from model_asif import AsifModel, IdentityClassifier
from noisy_label_dataset import create_noisy_dataset, NoisyDatasetType, NoisyDataset, parse_noisy_dataset_type
from optimizer_options import OptimizerOption, SchedulerOption, LossOption
from performance_data import PerformanceData
from phuber_losses import PHuberCrossEntropy
from sorting_memory_bank import SortingMemoryBank


class TrainerConfig:
    """Details for one particular experimental configuration."""

    def __init__(self,
                 lr: float,
                 config_name: str,
                 label: str,
                 description: str,
                 class_pick_clean_indices: CleanSamplePickingOption,
                 use_asif: bool,
                 use_cc: bool,
                 use_gce: bool,
                 use_phuber: bool,
                 cc_handle_positive_cases: bool,
                 cc_handle_negative_cases: bool,
                 cc_pick_clean_indices: bool,
                 cc_start_epoch: int,
                 cc_loss_coefficient: float,
                 optimizer_fe: OptimizerOption,
                 optimizer_if: OptimizerOption,
                 loss: LossOption,
                 scheduler: SchedulerOption,
                 nishi_lr_switch_iteration: int,
                 total_iterations: int,
                 true_label_estimate_loss_cutoff: int,
                 asif_start_epoch: int,
                 asif_loss_coefficient: int,
                 base_model_type: str,
                 shared_if_head_layer_count: int,
                 overridden_feature_vector_size: Optional[int],
                 if_reverse_private: bool,
                 if_reverse_public: bool,
                 class_pick_clean_indices_start_epoch: int,
                 class_pick_clean_indices_start_percentile: float,
                 class_pick_clean_indices_stop_percentile: float,
                 bad_label_picking_frequency: int,
                 bad_label_picking_sample_count: int,
                 bad_label_picking_cycle_count: int,
                 use_dgr: bool,
                 gce_q: float,
                 gce_start_prune_epoch: int,
                 sgd_decay: float,
                 sgd_momentum: float,
                 phuber_tau: float):
        """Create a new configuration.

        :argument lr The learning rate.
        :argument config_name A short name for inclusion in file names, etc.
        :argument label A human-readable name for clarity
        :argument description Longer description of configuration.
        :argument class_pick_clean_indices Train classifier with clean index picks by the method specified here.
        :argument use_asif If true, train with ASIF.
        :argument use_cc If true, train with contrastive clustering.
        :argument use_gce: If true, train with Generalised Cross Entropy Loss.
        :argument use_phuber: If true, train with partially Huberised (PHuber) cross-entropy loss.
        :argument phuber_tau: Tau hyperparameter used with partially Huberised (PHuber) cross-entropy loss.
        :argument cc_handle_positive_cases If true, apply CC to attract normal samples to their class centroid.
        :argument cc_handle_negative_cases If true, apply CC to repulse samples identified as having wrong labels away
        from their supposed class' centroid.
        :argument cc_pick_clean_indices if true, attempt to only apply CC to the clean samples.
        :argument cc_start_epoch If using constrastive clustering, only do so after this epoch.
        :argument cc_loss_coefficient Multiply the Contrastive Clustering loss term by this value.
        :argument optimizer_fe The type of optimiser to train the feature extractor with.
        :argument optimizer_if The type of optimiser to train the identity feature classifier with.
        :argument loss The loss to apply to the the classification.
        :argument scheduler The type of scheduler to train with. Options are 'Common', 'IDN', 'Nishi' and 'None'.
        :argument nishi_lr_switch_iteration The iteration after which to reduce the LR when using 'Nishi' scheduler.
        :argument total_iterations The total number of iterations to train for.
        :argument true_label_estimate_loss_cutoff If estimating whether a label is accurate, check if the loss is in the
        bottom `true_label_estimate_loss_cutoff` for its class.
        :argument base_model_type The type of feature extractor to use.
        :argument asif_start_epoch If using ASIF, start using ASIF on this epoch.
        :argument asif_loss_coefficient Coefficient to apply to ASIF loss.
        :argument shared_if_head_layer_count: The number of layers of the IF head that are shared across all classes.
        :argument overridden_feature_vector_size: If not None, reduce the dimensionality of the vector to be passed into
        the private IF heads.
        :argument if_reverse_private: Preform gradient reversal before ASIF private IF heads?
        :argument if_reverse_public: Preform gradient reversal before ASIF shared IF heads?
        :argument class_pick_clean_indices_start_epoch: The epoch during which the clean samples are picked. During each
        subsequent epoch, only those samples are trained.
        :argument class_pick_clean_indices_start_percentile: If sampling based on an IF metric, such as energy, only
        select samples that that fall in this percentile or higher.
        :argument class_pick_clean_indices_stop_percentile: If sampling based on an IF metric, such as energy, only
        select samples that that fall in this percentile or lower.
        :argument bad_label_picking_frequency: Perform picking of incorrectly labelled samples every N epochs. Picking
        is done via a combination of ASIF and classification loss.
        :argument bad_label_picking_sample_count: The number of samples to pick with each picking cycle. Picking is done
        via a combination of ASIF and classification loss. The number may be reduced in later cycles.
        :argument bad_label_picking_cycle_count: The total number of picking cycles to perform. Picking is done via a
        combination of ASIF and classification loss. After all picking cycles, training will proceed as normal.
        :argument use_dgr: If true, train ASIF using dynamic gradient reversal. Otherwise, use DANN.
        :argument gce_q: `q` parameter in GCE loss. See `GCELoss` class for more details.
        :argument gce_start_prune_epoch: The first epoch to begin pruning during GCE training.
        :argument sgd_decay: Decay rate if using SGD
        :argument sgd_momentum: Momentum if using SGD
        """
        self.lr = lr
        self.original_lr = lr
        self.config_name = config_name
        self.label = label
        self.description = description
        self.class_pick_clean_indices = class_pick_clean_indices
        self.use_ce = not use_gce and not use_phuber
        self.use_asif = use_asif
        self.use_cc = use_cc
        self.use_gce = use_gce
        self.use_phuber = use_phuber
        self.phuber_tau = phuber_tau
        self.cc_handle_positive_cases = cc_handle_positive_cases
        self.cc_handle_negative_cases = cc_handle_negative_cases
        self.cc_pick_clean_indices = cc_pick_clean_indices
        self.optimizer_fe = optimizer_fe
        self.optimizer_if = optimizer_if
        self.loss = loss
        self.scheduler = scheduler
        self.nishi_lr_switch_iteration = nishi_lr_switch_iteration
        self.total_iterations = total_iterations
        self.cc_start_epoch = cc_start_epoch
        self.cc_loss_coefficient = cc_loss_coefficient
        self.true_label_estimate_loss_cutoff = true_label_estimate_loss_cutoff
        self.base_model_type = base_model_type
        self.loss_collection_epoch = class_pick_clean_indices_start_epoch
        self.energy_collection_epoch = 1
        self.asif_start_epoch = asif_start_epoch
        self.asif_loss_coefficient = asif_loss_coefficient
        self.shared_if_head_layer_count = shared_if_head_layer_count
        self.overridden_feature_vector_size = overridden_feature_vector_size
        self.if_reverse_private = if_reverse_private
        self.if_reverse_public = if_reverse_public
        self.class_pick_clean_indices_start_epoch = class_pick_clean_indices_start_epoch
        self.class_pick_clean_indices_start_percentile = class_pick_clean_indices_start_percentile
        self.class_pick_clean_indices_stop_percentile = class_pick_clean_indices_stop_percentile
        self.use_dgr = use_dgr
        self.gce_q = gce_q
        self.gce_start_prune_epoch = gce_start_prune_epoch
        self.sgd_decay = sgd_decay
        self.sgd_momentum = sgd_momentum

        self.bad_label_picking_frequency = bad_label_picking_frequency
        self.bad_label_picking_sample_count = bad_label_picking_sample_count
        self.bad_label_picking_cycle_count = bad_label_picking_cycle_count


class TrainerASIFLabelNoise:
    def __init__(self,
                 data_dir: str,
                 dataset_type: NoisyDatasetType,
                 remediation_file: Optional[str],
                 configurations: Dict[str, TrainerConfig],
                 noise_method: LabelNoiseMethod,
                 noise_level: int,
                 noise_level_index: int,
                 samples_per_label: int,
                 apply_data_augmentation: bool,
                 label_file_dir: str,
                 use_unsupervised_asif: bool,
                 use_cuda: bool = True,
                 validate_freq: int = 5 * 391,
                 num_trials: int = 1,
                 first_trial_index: int = 0,
                 output_charts: bool = False,
                 output_tables: bool = False):
        """
        Creates a new ASIF trainer.
        :param data_dir: The path to the dataset.
        :param dataset_type: The type of dataset to create.
        :param remediation_file: If provided, this file contains suspected incorrect labels.
        :param configurations: A list of experimental configurations to run in parallel.
        :param noise_method: The method of label noise to use.
        :param noise_level: The level of label noise to apply. [0-100]
        :param noise_level_index: When multiple label noise files with the same method and level are available, which
        one do we choose? Useful for multiple trials with noise randomness.
        :param samples_per_label: The number of samples per label to train on.
        :param apply_data_augmentation: If true, apply data augmentation to dataset to reduce overfitting.
        :param label_file_dir: The path to the label noise files.
        :param use_unsupervised_asif: If true, use a single group for ASIF and run it even without labels.
        :param use_cuda: Train with CUDA?
        :param validate_freq: After this many epochs, run an accuracy check on the validation set.
        :param num_trials: The number of simultaneous trials of each configuration to run in parallel.
        :param first_trial_index: The index of the first trial for file naming, etc.
        :param output_charts: If true, output diagnostic visualisations.
        :param output_tables: If true, output diagnostic tables.
        """
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.num_trials = num_trials
        self.remediation_file = remediation_file
        self.use_remediation = remediation_file is not None
        self.first_trial_index = first_trial_index
        self.configurations = configurations
        self.noise_method = noise_method
        self.noise_level = noise_level
        self.noise_level_index = noise_level_index
        self.apply_data_augmentation = apply_data_augmentation
        self.label_file_dir = label_file_dir
        self.samples_per_label = samples_per_label
        self.use_unsupervised_asif = use_unsupervised_asif
        self.start_epoch = 0
        self.current_iteration = 0
        self.total_iterations = max([configurations[_config].total_iterations for _config in configurations])
        self.batch_size = 128
        self.validate_freq = validate_freq
        if apply_data_augmentation:
            self.validate_freq = self.validate_freq * 2
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.output_charts = output_charts
        self.split_across_gpus = torch.cuda.device_count() > 1
        self.probably_false_sample_indices = []
        self.output_tables = output_tables

        print("Gathering training data and summarizing...")
        self.training_data = None  # type: NoisyDataset
        self.evaluation_data = None  # type: NoisyDataset
        self.create_training_datasets()
        self.testing_data = create_noisy_dataset(data_dir=self.data_dir,
                                                 dataset_type=self.dataset_type,
                                                 train=False)
        print("...Done.")

        self.group_criteria = self._create_group_criteria()
        self.asif_per_sample_criteria = nn.CrossEntropyLoss(reduction='none')

        self.trial_data = {}  # type: Dict[str, List[_TrialData]]
        for _config in configurations:
            self.trial_data[_config] = []
            for _trial in range(self.num_trials):
                self.trial_data[_config].append(_TrialData(
                    config_name=_config,
                    label=configurations[_config].label,
                    trial_index=_trial + first_trial_index,
                    parent=self))

                if self.trial_data[_config][-1].state_file_name:
                    self.load_trial_state(_config, _trial, self.trial_data[_config][-1].state_file_name)

        if self.use_cuda:
            self.to_cuda()

    def create_training_datasets(self):
        false_label_indices = set(index for index, _ in self.probably_false_sample_indices)
        self.training_data = create_noisy_dataset(data_dir=self.data_dir,
                                                  dataset_type=self.dataset_type,
                                                  train=True,
                                                  apply_data_augmentation=self.apply_data_augmentation,
                                                  label_file_dir=self.label_file_dir,
                                                  label_remediation_file=self.remediation_file,
                                                  probably_false_sample_indices=false_label_indices,
                                                  noise_method=self.noise_method,
                                                  noise_level=self.noise_level,
                                                  trial_index=self.noise_level_index,
                                                  use_single_group=self.use_unsupervised_asif,
                                                  return_unlabelled_samples=self.use_unsupervised_asif,
                                                  samples_per_label=self.samples_per_label)
        self.evaluation_data = create_noisy_dataset(data_dir=self.data_dir,
                                                    dataset_type=self.dataset_type,
                                                    train=True,
                                                    randomize=False,
                                                    apply_data_augmentation=self.apply_data_augmentation,
                                                    label_file_dir=self.label_file_dir,
                                                    label_remediation_file=self.remediation_file,
                                                    probably_false_sample_indices=false_label_indices,
                                                    noise_method=self.noise_method,
                                                    noise_level=self.noise_level,
                                                    trial_index=self.noise_level_index,
                                                    use_single_group=self.use_unsupervised_asif,
                                                    return_unlabelled_samples=self.use_unsupervised_asif,
                                                    samples_per_label=self.samples_per_label)

    def create_model(self, config: str) -> nn.Module:
        """Creates a model for a given configuration.

        :parameter config Configuration name.

        :returns The model to test.
        """

        # The identity feature classifiers must each produce an output equal to the number of unique samples in each
        # label...
        group_counts = \
            [self.training_data.samples_per_group[_group] for _group in self.training_data.samples_per_group]

        model = AsifModel(
            data_dir='./data',
            in_channels=self.training_data.feature_count,
            class_count=self.training_data.class_count,
            group_counts=group_counts,
            base_model_type=self.configurations[config].base_model_type,
            use_log_softmax=self.configurations[config].loss == LossOption.LOSS_NLL,
            shared_if_head_layer_count=self.configurations[config].shared_if_head_layer_count,
            feature_vector_size=self.configurations[config].overridden_feature_vector_size,
            reverse_shared_head_gradient=self.configurations[config].if_reverse_public,
            use_dgr=self.configurations[config].use_dgr,
            total_iterations=self.total_iterations)

        if self.split_across_gpus:
            model = nn.DataParallel(model)
        return model

    def create_if_heads(self, parent_model: AsifModel, config: str) -> List[nn.Module]:
        """Creates the Identity Feature classifier heads, if we are using ASIF in this configuration.
        :parameter parent_model The parent model that will feed data into these IF heads.
        :parameter config Configuration name.
        :returns A list of IF heads.
        """

        # The identity feature classifiers must each produce an output equal to the number of unique samples in each
        # label...
        group_counts = [self.training_data.samples_per_group[_group]
                        for _group in range(len(self.training_data.samples_per_group))]

        if isinstance(parent_model, nn.DataParallel):
            model = parent_model.module
        else:
            model = parent_model

        model.clear_private_if_heads()

        identity_feature_classifiers = []
        for i in range(len(group_counts)):
            identity_feature_classifier = IdentityClassifier(
                in_features=model.feature_vector_size,
                out_features=group_counts[i],
                shared_part=False,
                reverse_gradient=self.configurations[config].if_reverse_private,
                layer_count=3 - self.configurations[config].shared_if_head_layer_count,
                use_dgr=self.configurations[config].use_dgr,
                total_iterations=self.total_iterations)
            identity_feature_classifiers.append(identity_feature_classifier)
            model.register_private_if_head(identity_feature_classifier)
        return identity_feature_classifiers

    def get_gpu_for_if_head(self, config_name: str, if_head_index: int) -> device:
        """
        Returns the GPU that an IF head is allocated to.
        :param config_name: The name of the current configuration.
        :param if_head_index: The index of the IF head in question.
        :return: A GPU device.
        """
        if self.split_across_gpus:
            gpu_count = torch.cuda.device_count()
            if_heads_per_gpu = len(self.trial_data[config_name][0].if_heads) // gpu_count
            gpu_index = if_head_index // if_heads_per_gpu
            if gpu_index >= gpu_count:
                gpu_index = gpu_count - 1

            return torch.device(f'cuda:{gpu_index}')
        else:
            return torch.device('cuda:0')

    def to_cuda(self):
        """Sends the components of the trainer to cuda."""
        for _config in self.trial_data:
            for trial_data in self.trial_data[_config]:
                trial_data.model = trial_data.model.cuda()
                trial_data.classification_criteria = trial_data.classification_criteria.cuda()
                trial_data.classification_per_sample_criteria = trial_data.classification_per_sample_criteria.cuda()

                if self.configurations[_config].use_gce:
                    trial_data.gce_criteria = trial_data.gce_criteria.cuda()

                if self.configurations[_config].use_phuber:
                    trial_data.phuber_criteria = trial_data.phuber_criteria.cuda()

                if self.configurations[_config].use_asif:
                    for i in range(len(trial_data.if_heads)):
                        trial_data.if_heads[i] = trial_data.if_heads[i].cuda()
                if self.configurations[_config].use_asif:
                    if self.split_across_gpus:
                        for i in range(len(trial_data.if_heads)):
                            gpu = self.get_gpu_for_if_head(config_name=_config, if_head_index=i)
                            trial_data.if_heads[i] = trial_data.if_heads[i].to(gpu)
                    else:
                        for i in range(len(trial_data.if_heads)):
                            trial_data.if_heads[i] = trial_data.if_heads[i].cuda()

        if self.split_across_gpus:
            for i in range(torch.cuda.device_count()):
                self.group_criteria[i] = self.group_criteria[i].to(torch.device(f'cuda:{i}'))
        else:
            # All criteria on the same GPU...
            self.group_criteria[0] = self.group_criteria[0].cuda()

        self.asif_per_sample_criteria = self.asif_per_sample_criteria.cuda()

    def load_trial_state(self, config: str, trial_index: int, state_file_path: str):
        """Loads the state of a particular trial into the trainer.
        :argument config The configuration of the experiment to load.
        :argument trial_index Zero-based index of the trial state to be loaded.
        :argument state_file_path The path to the state file.
        """
        print("Checking if config {0}, trial {1}, {2} exists...".format(config, trial_index, state_file_path))
        if os.path.isfile(state_file_path):
            print("=> loading checkpoint '{}'".format(state_file_path))
            checkpoint = torch.load(state_file_path)
            self.start_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.trial_data[config][trial_index].best_accuracy = checkpoint['best_accuracy']

            self.trial_data[config][trial_index].model.load_state_dict(checkpoint['model'])

            for optimizer_name in self.trial_data[config][trial_index].optimizers:
                self.trial_data[config][trial_index].optimizers[optimizer_name].load_state_dict(
                    checkpoint[optimizer_name])

            print("=> loaded checkpoint '{}' (epoch {})".format(state_file_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(state_file_path))

    def run_training(self):
        """
        Runs training.
        """
        start_time = datetime.now()

        epoch = self.start_epoch
        self.adjust_lr(epoch)

        self._write('Batch Size: {0}'.format(self.batch_size))
        self._write('Split Across GPUs: {0}'.format(self.split_across_gpus))
        self._write('Total Training Iterations: {0}'.format(self.total_iterations))
        self._write('Trial Count: {0}'.format(self.num_trials))
        self._write('Dataset: {0}'.format(self.dataset_type.name))
        self._write('Noise Method: {0}'.format(self.noise_method.name))
        self._write('Use Label Remediation: {0}'.format(self.use_remediation))
        self._write('Noise Level: {0}%'.format(self.noise_level))
        self._write('Label Noise File Index: {0}'.format(self.noise_level_index))
        self._write('Use Data Augmentation: {0}'.format(self.apply_data_augmentation))
        self._write('Use Unsupervised ASIF: {0}'.format(self.use_unsupervised_asif))
        if self.samples_per_label == 0:
            self._write('Samples Per Label: All')
        else:
            self._write('Samples Per Label: {0}'.format(self.samples_per_label))

        for _config in self.configurations:
            self._write('Learning Rate: {0}'.format(self.configurations[_config].lr), config=_config)
            self._write('Label: {0}'.format(self.configurations[_config].label), config=_config)
            self._write('Description: {0}'.format(self.configurations[_config].description), config=_config)
            self._write('Use ASIF: {0}'.format(self.configurations[_config].use_asif), config=_config)
            self._write('ASIF Start Epoch: {0}'.format(self.configurations[_config].asif_start_epoch), config=_config)
            self._write('ASIF Loss Coefficient: {0}'.format(self.configurations[_config].asif_loss_coefficient),
                        config=_config)
            self._write('ASIF Shared IF Head Size: {0}'.format(self.configurations[_config].shared_if_head_layer_count),
                        config=_config)
            self._write('ASIF Gradient Reversal before private IF heads?: {0}'.format(
                self.configurations[_config].if_reverse_private), config=_config)
            self._write('ASIF Gradient Reversal Method: {0}'.format(
                'DGR' if self.configurations[_config].use_dgr else 'DANN'), config=_config)

            if self.configurations[_config].use_ce:
                self._write('Classification Loss: Cross Entropy', config=_config)
            elif self.configurations[_config].use_gce:
                self._write('Classification Loss: Generalized Cross Entropy', config=_config)
            elif self.configurations[_config].use_phuber:
                self._write('Classification Loss: partially Huberised (PHuber) Cross Entropy', config=_config)
                self._write('PHuber Tao Value: {0}'.format(self.configurations[_config].phuber_tau), config=_config)
            else:
                raise RuntimeError("Either CE, GCE or PHuber must be selected.")

            if self.configurations[_config].use_gce:
                self._write('GCE q coefficient: {0}'.format(self.configurations[_config].gce_q), config=_config)
                self._write('GCE start pruning epoch: {0}'.format(self.configurations[_config].gce_start_prune_epoch),
                            config=_config)
            else:
                self._write('GCE q coefficient: N/A', config=_config)
                self._write('GCE start pruning epoch: N/A', config=_config)

            self._write('Train Classifier on Clean Samples: {0}'.format(
                self.configurations[_config].class_pick_clean_indices.get_short_name()), config=_config)
            self._write('Select clean samples during epoch {0}'.format(
                self.configurations[_config].class_pick_clean_indices_start_epoch), config=_config)
            self._write('Select samples in percentile range {0} to {1}'.format(
                self.configurations[_config].class_pick_clean_indices_start_percentile,
                self.configurations[_config].class_pick_clean_indices_stop_percentile), config=_config)

            self._write('Pick bad labels every X epochs: {0}'.format(
                self.configurations[_config].bad_label_picking_frequency), config=_config)
            self._write('Pick how many bad labels each time: {0}'.format(
                self.configurations[_config].bad_label_picking_sample_count), config=_config)
            self._write('Pick bad labels this many times: {0}'.format(
                self.configurations[_config].bad_label_picking_cycle_count), config=_config)

            if self.configurations[_config].overridden_feature_vector_size is None:
                feature_vector_str = "N/A"
            else:
                feature_vector_str = str(self.configurations[_config].overridden_feature_vector_size)
            self._write('Reduce ASIF feature vector dimensionality to: {0}'.format(feature_vector_str), config=_config)

            self._write('Use Contrastive Clustering: {0}'.format(self.configurations[_config].use_cc), config=_config)
            self._write('Use Positive CC: {0}'.format(self.configurations[_config].cc_handle_positive_cases),
                        config=_config)
            self._write('Use Negative CC: {0}'.format(self.configurations[_config].cc_handle_negative_cases),
                        config=_config)

            self._write('Use Contrastive Clustering on Clean Samples: {0}'.format(
                self.configurations[_config].cc_pick_clean_indices), config=_config)
            self._write('Use Contrastive Clustering after this epoch: {0}'.format(
                self.configurations[_config].cc_start_epoch), config=_config)
            self._write('Contrastive Clustering loss coefficient {0}'.format(
                self.configurations[_config].cc_loss_coefficient), config=_config)
            self._write('True Label Estimate Loss Cut-off: {0}'.format(
                self.configurations[_config].true_label_estimate_loss_cutoff), config=_config)
            self._write('Feature Extractor Model: {0}'.format(self.configurations[_config].base_model_type),
                        config=_config)
            self._write('Loss Data Collection Epoch: {0}'.format(self.configurations[_config].loss_collection_epoch),
                        config=_config)
            self._write('ASIF Energy Data Collection Epoch: {0}'.format(
                self.configurations[_config].energy_collection_epoch), config=_config)
            self._write('IF Optimizer: {0}'.format(self.configurations[_config].optimizer_if.get_short_name()),
                        config=_config)
            self._write('FE Optimizer: {0}'.format(self.configurations[_config].optimizer_fe.get_short_name()),
                        config=_config)
            if self.configurations[_config].optimizer_fe == OptimizerOption.OPTIMIZER_SGD:
                self._write('SGD Decay Rate: {0}'.format(self.configurations[_config].sgd_decay),
                            config=_config)
                self._write('SGD Momentum: {0}'.format(self.configurations[_config].sgd_momentum),
                            config=_config)
            else:
                self._write('SGD Decay Rate: N/A', config=_config)
                self._write('SGD Momentum: N/A', config=_config)
            self._write('Loss: {0}'.format(self.configurations[_config].loss.get_short_name()), config=_config)
            self._write('Scheduler: {0}'.format(self.configurations[_config].scheduler.get_short_name()),
                        config=_config)
            self._write('Nishi Switch LR: {0}'.format(self.configurations[_config].nishi_lr_switch_iteration),
                        config=_config)
            self._write('Training Iterations: {0}'.format(self.configurations[_config].total_iterations),
                        config=_config)

            for _trial in range(self.num_trials):
                self._write('Current Trial: {0}'.format(
                    self.trial_data[_config][_trial].trial_index), config=_config, trial_index=_trial)

        next_validation_iteration = self.current_iteration + self.validate_freq
        while self._should_keep_training():
            self.train_epoch(epoch)
            self.adjust_lr(epoch)
            if self.current_iteration < next_validation_iteration and self._should_keep_training():
                epoch += 1
                continue

            next_validation_iteration = self.current_iteration + self.validate_freq
            self.eval_if_heads(epoch)
            accuracy_per_trial = self.validate_epoch(epoch)
            self._flush_log()

            for _config in self.trial_data:
                for _trial in range(self.num_trials):
                    if accuracy_per_trial[_config][_trial] > self.trial_data[_config][_trial].best_accuracy:
                        self.trial_data[_config][_trial].best_accuracy = accuracy_per_trial[_config][_trial]

                        # Save this, since it's the best accuracy so far...
                        output_path = self.trial_data[_config][_trial].log_file_path.replace('.txt', '_best.pth.tar')
                        self.save_trial_state(epoch, _config, _trial, output_path)

                    self._write(' * best Accuracy {accuracy:4f}'.format(
                        accuracy=self.trial_data[_config][_trial].best_accuracy), trial_index=_trial, config=_config)

            epoch += 1
            # TODO: REMVOE THIS
            end_time = datetime.now()
            print(f'Elapsed Time: {end_time - start_time}')
            self.maybe_adjust_gce_loss(epoch)

        end_time = datetime.now()
        self._write(f'Elapsed Time: {end_time - start_time}')
        self._flush_log()

    def maybe_adjust_gce_loss(self, epoch: int):
        """
        We must periodically adjust the weights in the GCE loss. This occurs if we are using GCE and the training is
        advanced enough. See the `GCELoss` class for more details.
        :param epoch: The current training epoch.
        """
        perform_adjustment = False
        for _config in self.trial_data:
            if self.configurations[_config].use_gce and epoch >= self.configurations[_config].gce_start_prune_epoch:
                perform_adjustment = True
                break

        if not perform_adjustment:
            return

        # Load the best weights and recreate the models at their previous best...
        models: Dict[str, List[nn.Module]] = {}
        for _config in self.trial_data:
            if not self.configurations[_config].use_gce or epoch < self.configurations[_config].gce_start_prune_epoch:
                continue

            models[_config] = []
            for _trial in range(self.num_trials):
                weights_path = self.trial_data[_config][_trial].log_file_path.replace('.txt', '_best.pth.tar')

                checkpoint = torch.load(weights_path)
                model_weights = checkpoint['model']
                model = self.create_model(_config)
                model.load_state_dict(model_weights)
                if self.use_cuda:
                    model = model.cuda()
                model.eval()
                models[_config].append(model)

        # Now run the models through the dataset...
        train_loader = torch.utils.data.DataLoader(self.training_data,
                                                   num_workers=1,
                                                   pin_memory=True,
                                                   shuffle=True,
                                                   batch_size=self.batch_size)

        start_time = datetime.now()
        for i, (samples, label_ids, _, _, _, sample_ids, _) in enumerate(train_loader):
            samples = samples.type(torch.float32)
            label_ids = label_ids.type(torch.long)
            sample_ids = sample_ids.type(torch.long)

            if self.use_cuda:
                samples = samples.cuda(non_blocking=True)
                label_ids = label_ids.cuda(non_blocking=True)
                sample_ids = sample_ids.cuda(non_blocking=True)

            for _config in self.trial_data:
                if not self.configurations[_config].use_gce or \
                        epoch < self.configurations[_config].gce_start_prune_epoch:
                    continue

                for _trial in range(self.num_trials):
                    logits, _, _ = self.trial_data[_config][_trial].model(samples)
                    self.trial_data[_config][_trial].gce_criteria.update_weight(logits, label_ids, sample_ids)

            gc.collect()
            torch.cuda.empty_cache()

        elapsed_time = datetime.now() - start_time
        log_msg = "Iteration {0} (Epoch {1}) Performed GCE Pruning, Time: {2}".format(
            self.current_iteration, epoch, elapsed_time)
        for _config in self.trial_data:
            if self.configurations[_config].use_gce and epoch >= self.configurations[_config].gce_start_prune_epoch:
                self._write(log_msg, config=_config)

    def train_epoch(self, epoch: int, if_heads_only: bool = False):
        """Performs a single epoch of training.
        :argument epoch The current epoch.
        :argument if_heads_only If true, only train the IF heads. The rest of the model will be frozen...
        """
        task_losses = {}  # type: Dict[str, List[Dict[str, AverageMeter]]]
        loss_stats = {}  # type: Dict[str, List[Dict[int, Dict[str, float]]]]
        for _config in self.trial_data:
            if self.current_iteration < self.configurations[_config].total_iterations:
                task_losses[_config] = []
                loss_stats[_config] = []
                for _trial in range(self.num_trials):
                    task_losses[_config].append({})
                    task_losses[_config][_trial]['apparent_accuracy'] = AverageMeter()
                    task_losses[_config][_trial]['true_accuracy'] = AverageMeter()
                    task_losses[_config][_trial]['cfc'] = AverageMeter()
                    task_losses[_config][_trial]['cfc_clean_lbl'] = AverageMeter()
                    task_losses[_config][_trial]['cfc_noisy_lbl'] = AverageMeter()
                    task_losses[_config][_trial]['cc'] = AverageMeter()
                    task_losses[_config][_trial]['sel_dirty'] = AverageMeter()

                    if self.output_charts:
                        task_losses[_config][_trial]['sel_dirty_asif_10'] = AverageMeter()
                        task_losses[_config][_trial]['sel_dirty_asif_100'] = AverageMeter()
                        task_losses[_config][_trial]['sel_dirty_asif_1000'] = AverageMeter()
                        task_losses[_config][_trial]['sel_dirty_combined'] = AverageMeter()
                        task_losses[_config][_trial]['sel_dirty_combined_count'] = AverageMeter(output_sum=True)
                        task_losses[_config][_trial]['sel_dirty_asif_rev_10'] = AverageMeter()
                        task_losses[_config][_trial]['sel_dirty_asif_rev_100'] = AverageMeter()
                        task_losses[_config][_trial]['sel_dirty_asif_rev_1000'] = AverageMeter()
                        task_losses[_config][_trial]['sel_dirty_rev_combined'] = AverageMeter()
                        task_losses[_config][_trial]['sel_dirty_rev_combined_count'] = AverageMeter(output_sum=True)

                    task_losses[_config][_trial]['ifc'] = AverageMeter()
                    task_losses[_config][_trial]['ifc_clean'] = AverageMeter()
                    task_losses[_config][_trial]['ifc_dirty'] = AverageMeter()
                    task_losses[_config][_trial]['ife'] = AverageMeter()
                    task_losses[_config][_trial]['ife_clean'] = AverageMeter()
                    task_losses[_config][_trial]['ife_dirty'] = AverageMeter()
                    task_losses[_config][_trial]['ifs'] = AverageMeter()
                    task_losses[_config][_trial]['ifs_clean'] = AverageMeter()
                    task_losses[_config][_trial]['ifs_dirty'] = AverageMeter()
                    task_losses[_config][_trial]['asif_sel_dirty'] = AverageMeter()
                    task_losses[_config][_trial]['if_entropy_sel_dirty_20'] = AverageMeter()
                    task_losses[_config][_trial]['if_loss_sel_dirty_20'] = AverageMeter()
                    task_losses[_config][_trial]['if_entropy_sel_dirty_40'] = AverageMeter()
                    task_losses[_config][_trial]['if_loss_sel_dirty_40'] = AverageMeter()
                    task_losses[_config][_trial]['if_entropy_sel_dirty_60'] = AverageMeter()
                    task_losses[_config][_trial]['if_loss_sel_dirty_60'] = AverageMeter()
                    task_losses[_config][_trial]['if_entropy_sel_dirty_80'] = AverageMeter()
                    task_losses[_config][_trial]['if_loss_sel_dirty_80'] = AverageMeter()
                    for i in range(len(self.training_data.samples_per_group)):
                        task_losses[_config][_trial]['ifc{0}'.format(i)] = AverageMeter()
                        task_losses[_config][_trial]['ifc{0}_clean'.format(i)] = AverageMeter()
                        task_losses[_config][_trial]['ifc{0}_dirty'.format(i)] = AverageMeter()
                        task_losses[_config][_trial]['ife{0}'.format(i)] = AverageMeter()
                        task_losses[_config][_trial]['ife{0}_clean'.format(i)] = AverageMeter()
                        task_losses[_config][_trial]['ife{0}_dirty'.format(i)] = AverageMeter()
                        task_losses[_config][_trial]['ifs{0}'.format(i)] = AverageMeter()
                        task_losses[_config][_trial]['ifs{0}_clean'.format(i)] = AverageMeter()
                        task_losses[_config][_trial]['ifs{0}_dirty'.format(i)] = AverageMeter()

                    self.trial_data[_config][_trial].model.train()
                    for head in self.trial_data[_config][_trial].if_heads:
                        head.train()
                    self.trial_data[_config][_trial].create_optimizers(epoch=epoch)

                    # If this is IF head only, freeze the rest of the model...
                    for param in self.trial_data[_config][_trial].model.parameters():
                        param.requires_grad = not if_heads_only

                    loss_stats[_config].append({})

        train_loader = torch.utils.data.DataLoader(self.training_data,
                                                   num_workers=1,
                                                   pin_memory=True,
                                                   shuffle=True,
                                                   batch_size=self.batch_size)

        start_time = datetime.now()
        for i, (sample, label_id, true_label_id, group_id, group_membership_id, sample_id, probably_wrong) in \
                enumerate(train_loader):
            self._train_iter(epoch,
                             sample,
                             label_id,
                             true_label_id,
                             group_id,
                             group_membership_id,
                             sample_id,
                             probably_wrong,
                             task_losses,
                             loss_stats,
                             if_heads_only)

            gc.collect()
            torch.cuda.empty_cache()

            if not if_heads_only:
                # An IF heads only run doesn't count toward the iteration count...
                self.current_iteration += 1
        elapsed_time = datetime.now() - start_time

        for _config in self.configurations:
            for _trial in range(self.num_trials):
                trial_data = self.trial_data[_config][_trial]
                trial_data.loss_memory_bank_classification.build_lowest_first_measurement_cache()
                trial_data.loss_memory_bank_classification.build_highest_first_measurement_cache()
                trial_data.loss_memory_bank_asif.build_lowest_first_measurement_cache()
                trial_data.loss_memory_bank_asif.build_highest_first_measurement_cache()
                trial_data.asif_energy_memory_bank.build_lowest_first_measurement_cache()
                trial_data.asif_energy_memory_bank.build_highest_first_measurement_cache()

                for bank_ in trial_data.if_loss_memory_banks:
                    bank_.build_lowest_first_measurement_cache()
                    bank_.build_highest_first_measurement_cache()
                for bank_ in trial_data.if_entropy_memory_banks:
                    bank_.build_lowest_first_measurement_cache()
                    bank_.build_highest_first_measurement_cache()

                log_msg = 'Iteration {0} (Epoch {1})'.format(self.current_iteration, epoch)
                for loss_name in task_losses[_config][_trial]:
                    log_msg += ", {0}: {loss:.4f}".format(
                        loss_name,
                        loss=task_losses[_config][_trial][loss_name].get_final_value())

                if self.output_charts:
                    stats = self._calc_loss_stats(loss_stats[_config][_trial])
                    for i, (right_mean, right_stdev, wrong_mean, wrong_stdev) in enumerate(stats):
                        log_msg += ", IF {0} Right: {1} +/- {2}, IF {0} Wrong: {3} +/- {4}".format(i,
                                                                                                   right_mean,
                                                                                                   right_stdev,
                                                                                                   wrong_mean,
                                                                                                   wrong_stdev)

                log_msg += ", Time: {0}".format(elapsed_time)

                self._write(log_msg, trial_index=_trial, config=_config)

                if self.output_charts:
                    self._write_incorrect_selection_data_chart(epoch, _trial, _config)
                    self._write_correct_selection_data_chart(0, epoch, _trial, _config)
                    self._write_correct_selection_data_chart(1, epoch, _trial, _config)
                    self._write_correct_selection_data_chart(2, epoch, _trial, _config)
                    self._write_sample_data_histogram(epoch,
                                                      loss_stats[_config][_trial],
                                                      trial_index=_trial,
                                                      config=_config)

    def eval_if_heads(self, epoch: int):
        """Evaluates the IF heads' performance in regard to detecting dirty labels.
        :argument epoch The current epoch.
        """
        if self.noise_method == LabelNoiseMethod.LABEL_NOISE_NONE:
            return

        task_losses = {}  # type: Dict[str, List[Dict[str, AverageMeter]]]
        for _config in self.trial_data:
            task_losses[_config] = []
            for _trial in range(self.num_trials):
                task_losses[_config].append(
                    {'apparent_accuracy': AverageMeter(), 'true_accuracy': AverageMeter(),
                     'cls': AverageMeter(), 'cls_clean': AverageMeter(), 'cls_dirty': AverageMeter(),
                     'clss': AverageMeter(), 'clss_clean': AverageMeter(), 'clss_dirty': AverageMeter(),
                     'asif': AverageMeter(), 'asif_clean': AverageMeter(), 'asif_dirty': AverageMeter(),
                     'ife': AverageMeter(), 'ife_clean': AverageMeter(), 'ife_dirty': AverageMeter(),
                     'ifs': AverageMeter(), 'ifs_clean': AverageMeter(), 'ifs_dirty': AverageMeter(),
                     'ifean': AverageMeter(), 'ifean_clean': AverageMeter(), 'ifean_dirty': AverageMeter(),
                     'ifecn': AverageMeter(), 'ifeminn': AverageMeter(), 'ifemaxn': AverageMeter(),
                     'ifei': AverageMeter(), 'ifei_clean': AverageMeter(), 'ifei_dirty': AverageMeter(),
                     'ifsn': AverageMeter(), 'ifsn_clean': AverageMeter(), 'ifsn_dirty': AverageMeter(),
                     'ifsan': AverageMeter(), 'ifsan_clean': AverageMeter(), 'ifsan_dirty': AverageMeter(),
                     'ifscn': AverageMeter(), 'ifsminn': AverageMeter(), 'ifsmaxn': AverageMeter(),
                     'ifsi': AverageMeter(), 'ifsi_clean': AverageMeter(), 'ifsi_dirty': AverageMeter()})

                self.trial_data[_config][_trial].cls_memory_bank.start_new_epoch(epoch)
                self.trial_data[_config][_trial].cls_entropy_memory_bank.start_new_epoch(epoch)
                self.trial_data[_config][_trial].asif_loss_memory_bank.start_new_epoch(epoch)
                self.trial_data[_config][_trial].entropy_head_memory_bank.start_new_epoch(epoch)
                self.trial_data[_config][_trial].entropy_head_samplewise_memory_bank.start_new_epoch(epoch)
                self.trial_data[_config][_trial].energy_head_samplewise_memory_bank.start_new_epoch(epoch)

                self.trial_data[_config][_trial].model.eval()
                for head in self.trial_data[_config][_trial].if_heads:
                    head.eval()

        train_loader = torch.utils.data.DataLoader(self.evaluation_data,
                                                   num_workers=1,
                                                   pin_memory=True,
                                                   shuffle=False,
                                                   batch_size=self.batch_size)

        start_time = datetime.now()
        for i, (sample, label_id, true_label_id, group_ids, group_membership_ids, sample_id, _) in \
                enumerate(train_loader):
            self._eval_if_heads_iter(epoch,
                                     sample,
                                     label_id,
                                     true_label_id,
                                     group_ids,
                                     group_membership_ids,
                                     sample_id,
                                     task_losses)
        elapsed_time = datetime.now() - start_time

        # Set up the memory banks to see which sample picks are cleanest...
        for _config in self.configurations:
            for _trial in range(self.num_trials):
                trial_data = self.trial_data[_config][_trial]

                at_end = ['cls', 'cls_clean', 'cls_dirty',
                          'clss', 'clss_clean', 'clss_dirty',
                          'asif', 'asif_clean', 'asif_dirty',
                          'ifean', 'ifean_clean', 'ifean_dirty',
                          'ifsan', 'ifsan_clean', 'ifsan_dirty',
                          'ifsn', 'ifsn_clean', 'ifsn_dirty']
                log_msg = 'Eval IF Heads (Epoch {0})'.format(epoch)
                for loss_name in task_losses[_config][_trial]:
                    if loss_name not in at_end:
                        log_msg += ", {0}: {loss:.4f}".format(
                            loss_name,
                            loss=task_losses[_config][_trial][loss_name].get_final_value())

                for loss_name in at_end:
                    log_msg += ", {0}: {spread}".format(
                        loss_name,
                        spread=task_losses[_config][_trial][loss_name].get_spread())
                log_msg += f', Elapsed Time: {elapsed_time}'
                self._write(log_msg, trial_index=_trial, config=_config)

                trial_data.cls_memory_bank.normalise_latest_measurements()
                trial_data.cls_memory_bank.build_lowest_first_measurement_cache()
                trial_data.cls_memory_bank.build_highest_first_measurement_cache()
                trial_data.cls_entropy_memory_bank.normalise_latest_measurements()
                trial_data.cls_entropy_memory_bank.build_lowest_first_measurement_cache()
                trial_data.cls_entropy_memory_bank.build_highest_first_measurement_cache()
                trial_data.asif_loss_memory_bank.normalise_latest_measurements_by_label()
                trial_data.asif_loss_memory_bank.build_lowest_first_measurement_cache()
                trial_data.asif_loss_memory_bank.build_highest_first_measurement_cache()
                trial_data.entropy_head_memory_bank.normalise_latest_measurements_by_label()
                trial_data.entropy_head_memory_bank.build_lowest_first_measurement_cache()
                trial_data.entropy_head_samplewise_memory_bank.normalise_latest_measurements_by_label()
                trial_data.entropy_head_samplewise_memory_bank.build_lowest_first_measurement_cache()
                trial_data.energy_head_samplewise_memory_bank.normalise_latest_measurements_by_label()
                trial_data.energy_head_samplewise_memory_bank.build_lowest_first_measurement_cache()

                # Write samples...
                with open(trial_data.log_file_path.replace('.txt', f'_samples_{epoch}.csv'), 'w') as outfile:
                    outfile.write(
                        "Sample,Apparent Label,True Label,Classification Loss,Classification Entropy,ASIF Loss," +
                        "IF Head Entropy,Sample-wise IF Head Entropy,Sample-wise IF Head Energy,Combined\n")
                    for i in range(len(self.training_data)):
                        line = f"{i}"
                        line += f",{trial_data.cls_memory_bank.sample_labels[i].item()}"
                        line += f",{trial_data.cls_memory_bank.sample_true_labels[i].item()}"
                        line += f",{trial_data.cls_memory_bank.sample_metric[-1][i].item()}"
                        line += f",{trial_data.cls_entropy_memory_bank.sample_metric[-1][i].item()}"
                        line += f",{trial_data.asif_loss_memory_bank.sample_metric[-1][i].item()}"
                        line += f",{trial_data.entropy_head_memory_bank.sample_metric[-1][i].item()}"
                        line += f",{trial_data.entropy_head_samplewise_memory_bank.sample_metric[-1][i].item()}"
                        line += f",{trial_data.energy_head_samplewise_memory_bank.sample_metric[-1][i].item()}"

                        combined = trial_data.cls_memory_bank.sample_metric[-1][i].item() - \
                            trial_data.energy_head_samplewise_memory_bank.sample_metric[-1][i].item()
                        line += f",{combined}\n"
                        outfile.write(line)

                if self.output_charts:
                    # Output histogram data...
                    self._output_histogram(
                        epoch=epoch,
                        config=_config,
                        trial_index=_trial,
                        memory_bank=trial_data.cls_memory_bank,
                        title='Cls Loss Histogram (All)',
                        file_name='cls_histogram_all')

                    # Output histogram data...
                    self._output_histogram(
                        epoch=epoch,
                        config=_config,
                        trial_index=_trial,
                        memory_bank=trial_data.cls_entropy_memory_bank,
                        title='Cls Entropy Histogram (All)',
                        file_name='cls_entropy_histogram_all')

                    if self.configurations[_config].use_asif:
                        self._output_histogram(
                            epoch=epoch,
                            config=_config,
                            trial_index=_trial,
                            memory_bank=trial_data.asif_loss_memory_bank,
                            title='ASIF Loss Histogram (All)',
                            file_name='asif_loss_histogram_all')
                        self._output_histogram(
                            epoch=epoch,
                            config=_config,
                            trial_index=_trial,
                            memory_bank=trial_data.entropy_head_memory_bank,
                            title='ASIF Logit Entropy Histogram (All)',
                            file_name='asif_entropy_histogram_all')
                        self._output_histogram(
                            epoch=epoch,
                            config=_config,
                            trial_index=_trial,
                            memory_bank=trial_data.entropy_head_samplewise_memory_bank,
                            title='ASIF Logit Samplewise Entropy Histogram (All)',
                            file_name='asif_entropy_samplewise_histogram_all')
                        self._output_histogram(
                            epoch=epoch,
                            config=_config,
                            trial_index=_trial,
                            memory_bank=trial_data.energy_head_samplewise_memory_bank,
                            title='ASIF Logit Samplewise Energy Histogram (All)',
                            file_name='asif_energy_samplewise_histogram_all')

                # Now let's pick some dirty labels and then reset the IF heads...
                self._reinitialise_if_heads(epoch, _config, _trial)

    def _reinitialise_if_heads(self, epoch: int, config: str, trial_index: int):
        """
        Pick some dirty labels and then reinitialise the private IF heads to account for them.
        :param epoch: The current epoch
        :param config: The current configuration
        :param trial_index: The current trial
        """
        config_data = self.configurations[config]

        if config_data.bad_label_picking_cycle_count == 0:
            return

        if (epoch + 1) % config_data.bad_label_picking_frequency != 0:
            return

        # Only do this a certain number of times...
        if epoch > config_data.bad_label_picking_frequency * config_data.bad_label_picking_cycle_count:
            return

        # First pick some dirty labels...
        self._find_dirty(epoch, config, trial_index)

        # Reset the trial data. The dirty sample data will be included in the recreation, which will affect the IF head
        # allocations...
        self.create_training_datasets()

        # Completely reset the model...
        if self.configurations[config].use_asif:
            self.trial_data[config][trial_index].reset_trainer()
            if self.use_cuda:
                self.to_cuda()

        # Set epoch to zero because we want to keep the default LR...
        self.trial_data[config][trial_index].create_optimizers(epoch=0, force_creation=True)

        # Reset the memory banks...
        self.trial_data[config][trial_index].cls_memory_bank = SortingMemoryBank(
            dataset_size=len(self.training_data),
            history_length=3,
            history_forward=False)
        self.trial_data[config][trial_index].cls_entropy_memory_bank = SortingMemoryBank(
            dataset_size=len(self.training_data),
            history_length=3,
            history_forward=False)
        self.trial_data[config][trial_index].asif_loss_memory_bank = SortingMemoryBank(
            dataset_size=len(self.training_data),
            history_length=3,
            history_forward=False)

    def _is_probably_false_label(self, sample_index: int) -> bool:
        """
        If the trainer determines that a sample with a given `index`'s label is probably false, this will return 'True'.
        :param sample_index: The index of the sample in question.
        """
        for sample_index_, _ in self.probably_false_sample_indices:
            if sample_index_ == sample_index:
                return True

        return False

    def _mark_as_probably_false_label(self, sample_index: int, apparent_label: int):
        """
        If the trainer determines that a sample with a given `index`'s label is probably false, it can call this method.
        In future iterations, the `is_probably_incorrect_label` value returned from the training data will be '1'.
        :param sample_index: The index of the sample in question.
        :param apparent_label: The apparent label of the sample.
        """
        self.probably_false_sample_indices.append((sample_index, apparent_label))
        self.training_data.mark_as_probably_false_label(sample_index)

    def _test_find_dirty(self, config: str, trial_index: int):
        trial_data = self.trial_data[config][trial_index]

        items = []
        for i in range(trial_data.cls_memory_bank.sample_metric_mean.size()[0]):
            if self.configurations[config].use_asif:
                asif_loss = trial_data.asif_loss_memory_bank.sample_metric[-1][i].item()
                sample_label = trial_data.asif_loss_memory_bank.sample_labels[i].item()
            else:
                asif_loss = 0.
                sample_label = trial_data.cls_memory_bank.sample_labels[i].item()
            cls_loss = trial_data.cls_memory_bank.sample_metric[-1][i].item()
            cls_entropy = trial_data.cls_entropy_memory_bank.sample_metric[-1][i].item()
            if sample_label >= 0:
                items.append((i,
                              trial_data.cls_memory_bank.sample_labels[i].item(),
                              trial_data.cls_memory_bank.sample_true_labels[i].item(),
                              asif_loss - cls_loss - cls_entropy))
        items = sorted(items, key=lambda datum: datum[3], reverse=False)

        total_marked = 0
        total_incorrectly_marked = 0
        for i in range(len(items)):
            index_, label_, true_label_, _ = items[i]

            total_marked += 1
            if label_ == true_label_:
                total_incorrectly_marked += 1

            if total_marked in [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 20000, 30000]:
                msg = f"Test found {total_marked} samples falsely labeled."
                msg += f" ({total_incorrectly_marked} incorrectly - {total_incorrectly_marked * 100 / total_marked}%)"
                self._write(msg, config=config, trial_index=trial_index)

        msg = f"Test found {total_marked} samples falsely labeled."
        msg += f" ({total_incorrectly_marked} incorrectly - {total_incorrectly_marked * 100 / total_marked}%)"
        self._write(msg, config=config, trial_index=trial_index)

    def _find_dirty(self, epoch: int, config: str, trial_index: int):
        trial_data = self.trial_data[config][trial_index]

        selection = self.configurations[config].bad_label_picking_sample_count

        items = []
        for i in range(trial_data.cls_memory_bank.sample_metric_mean.size()[0]):
            sample_label = trial_data.cls_memory_bank.sample_labels[i].item()
            cls_loss = trial_data.cls_memory_bank.sample_metric[-1][i].item()
            if sample_label >= 0:
                items.append((i,
                              trial_data.cls_memory_bank.sample_labels[i].item(),
                              trial_data.cls_memory_bank.sample_true_labels[i].item(),
                              cls_loss))
        items = sorted(items, key=lambda datum: datum[3], reverse=True)

        # Loop until we've got enough...
        total_marked = 0
        total_incorrectly_marked = 0
        for i in range(len(items)):
            index_, label_, true_label_, _ = items[i]

            # If it is already marked false, don't do it again...
            if self.training_data.is_probably_false_label(index_):
                continue

            total_marked += 1
            if label_ == true_label_:
                total_incorrectly_marked += 1

            self._mark_as_probably_false_label(index_, label_)

            if total_marked >= selection:
                break

        self._write(f"Marked {total_marked} samples falsely labeled. ({total_incorrectly_marked} incorrectly)",
                    config=config, trial_index=trial_index)

        # Write the remediation file...
        file_path = trial_data.log_file_path.replace('.txt', f'_{epoch}_remediation_file.csv')
        with open(file_path, 'w') as outfile:
            for index, label in self.probably_false_sample_indices:
                outfile.write(f"{index},{label},True\n")

    def _output_energy_histogram(self,
                                 epoch: int,
                                 config: str,
                                 trial_index: int,
                                 true_energy_bank: SortingMemoryBank,
                                 apparent_energy_bank: SortingMemoryBank,
                                 bin_start: float,
                                 bin_stop: float,
                                 bin_count: int,
                                 title: str,
                                 file_name: str):
        """
        Creates a histogram of energy standard deviations between the outputs of the apparently-correct IF head and the
        overall spread of energy across all IF heads. Does the same with the actually correct IF head.
        :param epoch: The current epoch.
        :param config: The experiment configuration to save.
        :param trial_index: The index of the trial to save.
        :param true_energy_bank Contains the normalised energy for each sample from the IF head corresponding to the
        correct label, relative to the average output across all IF heads.
        :param apparent_energy_bank Contains the normalised energy for each sample from the IF head corresponding to the
        apparently correct label, relative to the average output across all IF heads.
        :param bin_start: First value for histogram bins.
        :param bin_stop: Final value for histogram bins.
        :param bin_count: The number of histogram bins.
        :param title: The title to add to the plot.
        :param file_name: The name of the file to output.
        """
        # noinspection PyUnresolvedReferences
        bins = np.linspace(bin_start, bin_stop, bin_count)

        true_data = []
        apparent_data = []
        for i in range(true_energy_bank.sample_metric_mean.size()[0]):
            # Only include the data if it corresponds to a sample that is mislabelled...
            if true_energy_bank.sample_labels[i] != true_energy_bank.sample_true_labels[i]:
                true_data.append(true_energy_bank.sample_metric_mean[i].item())
                apparent_data.append(apparent_energy_bank.sample_metric_mean[i].item())

        fig = plt.figure(figsize=(8, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title, fontsize=20)

        wrong_bars, _, _ = ax.hist(true_data, bins, alpha=0.5, label='Energy from True Heads')
        right_bars, _, _ = ax.hist(apparent_data, bins, alpha=0.5, label='Energy from Apparent Heads')
        ax.legend(loc='upper right')

        file_path = self.trial_data[config][trial_index].log_file_path.replace(
            ".txt",
            f"_{epoch}_{file_name}.png")
        fig.savefig(file_path)

    def _output_histogram(self,
                          epoch: int,
                          config: str,
                          trial_index: int,
                          memory_bank: SortingMemoryBank,
                          title: str,
                          file_name: str,
                          output_individuals: bool = False):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        if output_individuals:
            bin_size = 250
            class_count = len(self.training_data.samples_per_group)
            for class_index in range(class_count):
                energies, incorrect_labels, label_counts = memory_bank.get_histogram_data(filter_to_label=class_index,
                                                                                          bin_size=bin_size)
                y = []
                x = []
                for i in range(len(incorrect_labels)):
                    y.append(incorrect_labels[i] * 100. / label_counts[i])
                    x.append(i)

                ax.plot(x, y, label=f'% Dirty (IF Head {class_index})')
        else:
            bin_size = 2500
            energies, incorrect_labels, label_counts = memory_bank.get_histogram_data(bin_size=bin_size)
            y = []
            x = []
            for i in range(len(incorrect_labels)):
                y.append(incorrect_labels[i] * 100. / label_counts[i])
                x.append(i)

            ax.plot(x, y)

        file_path = self.trial_data[config][trial_index].log_file_path.replace(
            ".txt",
            f"_{epoch}_{file_name}.png")
        fig.savefig(file_path)

    def _output_energy_truefalse_histogram(self,
                                           epoch: int,
                                           config: str,
                                           trial_index: int,
                                           apparent_energy_bank: SortingMemoryBank,
                                           bin_start: float,
                                           bin_stop: float,
                                           bin_count: int,
                                           title: str,
                                           file_name: str):
        """
        Creates a histogram of energy standard deviations between the outputs of the apparently-correct IF head and the
        overall spread of energy across all IF heads. Separately shows the values for false labels.
        :param epoch: The current epoch.
        :param config: The experiment configuration to save.
        :param trial_index: The index of the trial to save.
        :param apparent_energy_bank Contains the normalised energy for each sample from the IF head corresponding to the
        apparently correct label, relative to the average output across all IF heads.
        :param bin_start: First value for histogram bins.
        :param bin_stop: Final value for histogram bins.
        :param bin_count: The number of histogram bins.
        :param title: The title to add to the plot.
        :param file_name: The name of the file to output.
        """
        # noinspection PyUnresolvedReferences
        bins = np.linspace(bin_start, bin_stop, bin_count)

        false_labelled_data = []
        true_labelled_data = []
        for i in range(apparent_energy_bank.sample_metric_mean.size()[0]):
            # Only include the data if it corresponds to a sample that is mislabelled...
            if apparent_energy_bank.sample_labels[i] != apparent_energy_bank.sample_true_labels[i]:
                false_labelled_data.append(apparent_energy_bank.sample_metric_mean[i].item())
            else:
                true_labelled_data.append(apparent_energy_bank.sample_metric_mean[i].item())

        fig = plt.figure(figsize=(8, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title, fontsize=20)

        wrong_bars, _, _ = ax.hist(false_labelled_data, bins, alpha=0.5, label='False Labelled Energy')
        right_bars, _, _ = ax.hist(true_labelled_data, bins, alpha=0.5, label='True Labelled Energy')
        ax.legend(loc='upper right')

        file_path = self.trial_data[config][trial_index].log_file_path.replace(
            ".txt",
            f"_{epoch}_{file_name}.png")
        fig.savefig(file_path)

    def validate_epoch(self, epoch: float) -> Dict[str, List[float]]:
        """Performs validation of the model after an epoch of training.

        Returns. The average balanced accuracy for each config and trial being run.
        """
        print('begin test')

        test_loader = torch.utils.data.DataLoader(self.testing_data,
                                                  pin_memory=True,
                                                  batch_size=self.batch_size,
                                                  num_workers=1)

        stats = {}  # type: Dict[str, List[PerformanceData]]
        accuracies = {}  # type: Dict[str, List[Accuracy]]
        for _config in self.trial_data:
            stats[_config] = []
            accuracies[_config] = []
            for trial_data in self.trial_data[_config]:
                trial_data.model.eval()
                for head in trial_data.if_heads:
                    head.eval()
                stats[_config].append(PerformanceData(self.testing_data.class_names))
                accuracies[_config].append(Accuracy())

        count = 0

        for i, (sample, _, true_label_id, _, _, _, _) in enumerate(test_loader):
            count += 1
            self._eval_iter(sample, true_label_id, stats, accuracies)

        balanced_accuracies = {}  # type: Dict[str, List[float]]
        for _config in self.trial_data:
            balanced_accuracies[_config] = []
            for _trial in range(self.num_trials):
                if self.output_tables:
                    self._write("\n" + stats[_config][_trial].get_summary_str(), trial_index=_trial, config=_config)

                balanced_accuracies[_config].append(accuracies[_config][_trial].result())

        self._write('COMPLETED Testing Epoch {0}.'.format(epoch))
        for _config in self.trial_data:
            for _trial in range(self.num_trials):
                self._write(' * Accuracy {balanced_accuracy:04f}'.format(
                    balanced_accuracy=balanced_accuracies[_config][_trial]), trial_index=_trial, config=_config)

            # Work out statistics...
            accuracy_mean = statistics.mean(balanced_accuracies[_config])
            if len(balanced_accuracies[_config]) > 1:
                accuracy_std = statistics.stdev(balanced_accuracies[_config])
            else:
                accuracy_std = 0.
            self._write(' * Accuracy Stats: {0} +/- {1} ({2})'.format(
                int(accuracy_mean * 1000) / 10,
                int(accuracy_std * 1000) / 10,
                len(balanced_accuracies[_config])), config=_config)

        return balanced_accuracies

    def _write_asif_data_tables(self,
                                epoch: int,
                                data: Dict[str, List[List[Dict[str, float]]]],
                                trial_index: int,
                                config: str):
        for datum in ['energy', 'entropy']:
            line = 'IF Head,'
            for _class_id in range(self.testing_data.class_count):
                line += f'Class {_class_id},'
            line += '\n'

            for _head_id in range(self.testing_data.class_count):
                line += f'{_head_id},'
                for _class_id in range(self.testing_data.class_count):
                    values = []
                    for _data in data[config][trial_index]:
                        if _data['correct_label'] == _class_id and _data['if_head_id'] == _head_id:
                            values.append(_data[datum])
                    data_mean = statistics.mean(values)
                    data_stdev = statistics.stdev(values)
                    line += f'{data_mean} +/- {data_stdev},'
                line += '\n'

            file_path = self.trial_data[config][trial_index].log_file_path.replace(
                ".txt",
                f"_asif_{datum}_{epoch}.csv")
            with open(file_path, 'w') as outfile:
                outfile.write(line)

    def _write_asif_data_histogram(self,
                                   epoch: int,
                                   data: Dict[str, List[List[Dict[str, float]]]],
                                   trial_index: int,
                                   config: str):
        """
        Takes a datum calculated for each individual sample and saves a histogram. The datum is specified by `datum`
        and can be found in the `data` structure.
        :param epoch: The current epoch.
        :param data: The data to plot.
        :param config: The experiment configuration to save.
        :param trial_index: The index of the trial to save.
        """
        _data = data[config][trial_index]

        for datum in ['energy', 'entropy']:
            fig = plt.figure(figsize=(20, 10))
            for _head_id in range(self.testing_data.class_count):
                data_to_plot = \
                    [(rec[datum], rec['correct_label']) for rec in _data if rec['if_head_id'] == _head_id]

                if datum == 'energy':
                    # noinspection PyUnresolvedReferences
                    bins = np.linspace(-30, 30, 100)
                else:
                    # noinspection PyUnresolvedReferences
                    bins = np.linspace(0, 8, 100)

                ax = fig.add_subplot(2, 5, _head_id + 1)
                ax.set_title(f'IF Head {_head_id} {datum} Distributions', fontsize=12)

                correct_data = [data_to_plot[i][0] for i in range(len(data_to_plot)) if data_to_plot[i][1] == _head_id]
                incorrect_data = \
                    [data_to_plot[i][0] for i in range(len(data_to_plot)) if data_to_plot[i][1] != _head_id]
                ax.hist(correct_data, bins, alpha=0.5, label='Correct')
                ax.hist(incorrect_data, bins, alpha=0.5, label='Incorrect')

                ax.legend(loc='upper right')

            file_path = self.trial_data[config][trial_index].log_file_path.replace(
                ".txt",
                f"_asif_{datum}_histogram_{epoch}.png")
            fig.savefig(file_path)

    def save_trial_state(self, epoch: int, config: str, trial_index: int, output_path: str):
        r"""Saves the current state of training.

        Args:
            epoch: The current epoch number
            config: The experiment configuration to save.
            trial_index: The index of the trial to save.
            output_path: The path to the file to save to.
        """
        state = {
            'epoch': epoch + 1,
            'iteration': self.current_iteration,
            'trial_index': self.trial_data[config][trial_index].trial_index,
            'config': config,
            'use_asif': self.configurations[config].use_asif,
            'use_cc': self.configurations[config].use_cc,
            'use_gce': self.configurations[config].use_gce,
            'use_phuber': self.configurations[config].use_phuber,
            'phuber_tau': self.configurations[config].phuber_tau,
            'use_ce': self.configurations[config].use_ce,
            'cc_handle_positive_cases': self.configurations[config].cc_handle_positive_cases,
            'cc_handle_negative_cases': self.configurations[config].cc_handle_negative_cases,
            'cc_pick_clean_indices': self.configurations[config].cc_pick_clean_indices,
            'class_pick_clean_indices': self.configurations[config].class_pick_clean_indices,
            'asif_start_epoch': self.configurations[config].asif_start_epoch,
            'asif_loss_coefficient': self.configurations[config].asif_loss_coefficient,
            'use_dgr': self.configurations[config].use_dgr,
            'gce_q': self.configurations[config].gce_q,
            'gce_start_prune_epoch': self.configurations[config].gce_start_prune_epoch,
            'shared_if_head_layer_count': self.configurations[config].shared_if_head_layer_count,
            'if_reverse_private': self.configurations[config].if_reverse_private,
            'if_reverse_public': self.configurations[config].if_reverse_public,
            'class_pick_clean_indices_start_epoch': self.configurations[config].class_pick_clean_indices_start_epoch,
            'class_pick_clean_indices_start_percentile':
                self.configurations[config].class_pick_clean_indices_start_percentile,
            'class_pick_clean_indices_stop_percentile':
                self.configurations[config].class_pick_clean_indices_stop_percentile,
            'overridden_feature_vector_size': self.configurations[config].overridden_feature_vector_size,
            'bad_label_picking_frequency': self.configurations[config].bad_label_picking_frequency,
            'bad_label_picking_sample_count': self.configurations[config].bad_label_picking_sample_count,
            'bad_label_picking_cycle_count': self.configurations[config].bad_label_picking_cycle_count,
            'cc_start_epoch': self.configurations[config].cc_start_epoch,
            'nishi_lr_switch_iteration': self.configurations[config].nishi_lr_switch_iteration,
            'total_iterations': self.configurations[config].total_iterations,
            'cc_loss_coefficient': self.configurations[config].cc_loss_coefficient,
            'true_label_estimate_loss_cutoff': self.configurations[config].true_label_estimate_loss_cutoff,
            'base_model_type': self.configurations[config].base_model_type,
            'loss_collection_epoch': self.configurations[config].loss_collection_epoch,
            'energy_collection_epoch': self.configurations[config].energy_collection_epoch,
            'noise_method': self.noise_method.name,
            'use_remediation': self.use_remediation,
            'dataset_type': self.dataset_type.name,
            'noise_level': self.noise_level,
            'noise_level_index': self.noise_level_index,
            'apply_data_augmentation': self.apply_data_augmentation,
            'samples_per_label': self.samples_per_label,
            'use_unsupervised_asif': self.use_unsupervised_asif,
            'label': self.trial_data[config][trial_index].label,
            'description': self.configurations[config].description,
            'optimizer_if': self.configurations[config].optimizer_if,
            'optimizer_fe': self.configurations[config].optimizer_fe,
            'sgd_decay': self.configurations[config].sgd_decay,
            'sgd_momentum': self.configurations[config].sgd_momentum,
            'loss': self.configurations[config].loss,
            'scheduler': self.configurations[config].scheduler,
            'best_accuracy': self.trial_data[config][trial_index].best_accuracy,
            'model': self.trial_data[config][trial_index].model.state_dict(),
            'batch_size': self.batch_size,
            'lr': self.configurations[config].lr,
            'samples_per_group': self.training_data.samples_per_group,
            'group_counts':
                [self.training_data.samples_per_group[_group] for _group in self.training_data.samples_per_group],
        }

        if_head_states = []
        for head_ in self.trial_data[config][trial_index].if_heads:
            if_head_states.append(head_.state_dict())
        state['if_heads'] = if_head_states

        for optimizer_name in self.trial_data[config][trial_index].optimizers:
            state[optimizer_name] = self.trial_data[config][trial_index].optimizers[optimizer_name].state_dict()

        torch.save(state, output_path)

    def adjust_lr(self, epoch: int):
        """Adjust learning rate."""
        if epoch == 0:
            return

        for _config in self.trial_data:
            for _trial in range(self.num_trials):
                prev_lr = self.configurations[_config].lr
                if self.configurations[_config].scheduler == SchedulerOption.SCHEDULER_COMMON_LEARNING:
                    adjustment_schedule = [19500, 25000, 30000]
                    self.configurations[_config].lr = self.configurations[_config].original_lr
                    for _item in adjustment_schedule:
                        if self.current_iteration > _item:
                            self.configurations[_config].lr *= 0.1

                    if self.configurations[_config].lr != prev_lr:
                        self._write('Adjusting original LR of {0} down to {1}.'.format(
                            self.configurations[_config].original_lr,
                            self.configurations[_config].lr))

                    for param_group in self.trial_data[_config][_trial].optimizers['model_optimizer'].param_groups:
                        param_group['lr'] = self.configurations[_config].lr

                elif self.configurations[_config].scheduler == SchedulerOption.SCHEDULER_IDN:
                    factor = 0
                    if epoch > 120:
                        factor = 2
                    elif epoch > 60:
                        factor = 1
                    self.configurations[_config].lr = self.configurations[_config].original_lr * math.pow(0.2, factor)

                    if self.configurations[_config].lr != prev_lr:
                        self._write('Adjusting original LR of {0} down to {1}.'.format(
                            self.configurations[_config].original_lr,
                            self.configurations[_config].lr))

                    for param_group in self.trial_data[_config][_trial].optimizers['model_optimizer'].param_groups:
                        param_group['lr'] = self.configurations[_config].lr

                elif self.configurations[_config].scheduler == SchedulerOption.SCHEDULER_GCE:
                    factor = 1
                    if epoch > 80:
                        factor = 0.01
                    elif epoch > 40:
                        factor = 0.1
                    self.configurations[_config].lr = self.configurations[_config].original_lr * factor

                    if self.configurations[_config].lr != prev_lr:
                        self._write('Adjusting original LR of {0} down to {1}.'.format(
                            self.configurations[_config].original_lr,
                            self.configurations[_config].lr))

                    for param_group in self.trial_data[_config][_trial].optimizers['model_optimizer'].param_groups:
                        param_group['lr'] = self.configurations[_config].lr

                elif self.configurations[_config].scheduler == SchedulerOption.SCHEDULER_PHUBER:
                    factor = 1
                    if epoch > 160:
                        factor = 0.2 * 0.2 * 0.2
                    elif epoch > 120:
                        factor = 0.2 * 0.2
                    elif epoch > 160:
                        factor = 0.2
                    self.configurations[_config].lr = self.configurations[_config].original_lr * factor

                    if self.configurations[_config].lr != prev_lr:
                        self._write('Adjusting original LR of {0} down to {1}.'.format(
                            self.configurations[_config].original_lr,
                            self.configurations[_config].lr))

                    for param_group in self.trial_data[_config][_trial].optimizers['model_optimizer'].param_groups:
                        param_group['lr'] = self.configurations[_config].lr

                elif self.configurations[_config].scheduler == SchedulerOption.SCHEDULER_NISHI:
                    self.configurations[_config].lr = self.configurations[_config].original_lr
                    if self.current_iteration > self.configurations[_config].nishi_lr_switch_iteration:
                        self.configurations[_config].lr *= 0.1

                    if self.configurations[_config].lr != prev_lr:
                        self._write('Adjusting original LR of {0} down to {1}.'.format(
                            self.configurations[_config].original_lr,
                            self.configurations[_config].lr))

                    for param_group in self.trial_data[_config][_trial].optimizers['model_optimizer'].param_groups:
                        param_group['lr'] = self.configurations[_config].lr

    def _train_task_cc(self,
                       epoch: int,
                       label_id: torch.Tensor,
                       probably_wrong: torch.Tensor,
                       feature_vectors: torch.Tensor,
                       labelled_indices: torch.Tensor,
                       config_name: str,
                       trial_index: int,
                       task_losses: Dict[str, List[Dict[str, AverageMeter]]]) -> torch.Tensor:
        """
        Performs the Contrastive Clustering task for training. Returns a loss if appropriate. Otherwise, returns None.
        :param epoch: The current training epoch.
        :param label_id: The labels for each sample in the mini-batch.
        :param probably_wrong: If 1, the label may not be correct.
        :param feature_vectors: The feature vectors output from the model's feature extractor.
        :param labelled_indices: The indices of the samples within the mini-batch that have labels.
        :param config_name: The name of the configuration being trained.
        :param trial_index: The trial index being trained.
        :param task_losses: Accumulates loss statistics.
        :return: The loss resulting from the task. Can be None.
        """
        probably_right = probably_wrong == 0

        if labelled_indices.sum().item() == 0:
            # No labelled data, skip...
            return None

        label_id_labelled = label_id[labelled_indices]
        feature_vectors_labelled = feature_vectors[labelled_indices]
        probably_right_labelled = probably_right[labelled_indices]

        if probably_right_labelled.sum().item() > 0:
            self.trial_data[config_name][trial_index].cc_manager.add_feature_vectors_to_history(
                feature_vectors_labelled[probably_right_labelled],
                label_id_labelled[probably_right_labelled],
                probably_right_labelled[probably_right_labelled])
        if epoch >= self.configurations[config_name].cc_start_epoch:
            cc_loss = self.trial_data[config_name][trial_index].cc_manager.calc_loss(
                feature_vectors=feature_vectors_labelled,
                labels=label_id_labelled,
                probably_right=probably_right_labelled,
                current_epoch=epoch)
            if cc_loss is not None:
                # Apply CC loss coefficient...
                cc_loss = cc_loss * self.configurations[config_name].cc_loss_coefficient

                task_losses[config_name][trial_index]['cc'].update(cc_loss.item())
                return cc_loss

        return None

    @staticmethod
    def _calc_dirty_marked_clean_percent(
            memory_bank: LossMemoryBank,
            sample_ids: torch.Tensor,
            incorrect_label_indices: torch.Tensor,
            cutoff: int,
            take_top: bool = False,
            take_bottom: bool = True) -> float:
        """
        Determines which of the samples specified by `sample_ids` are in the bottom `cutoff` samples, as determined
        by a given `memory_bank`. It then returns the percentage of those selected samples that have dirty labels.

        Note that `take_top` and `take_top` can be used simultaneously.

        :param memory_bank: The memory bank measuring sample data, such as loss or entropy.
        :param sample_ids: The IDs of the samples in the batch being measured.
        :param incorrect_label_indices: The indices of the elements within the batch that have incorrect labels.
        :param cutoff: Checks if the samples are in the bottom 'X' number of samples, as measured by the memory bank.
        :param take_top: If true, take the top X samples.
        :param take_bottom: If true, take the bottom X samples.
        :return: The percentage of the batch that is both part of the selected group and having incorrect labels.
        """
        if take_bottom and take_top:
            bottom_indices = memory_bank.is_in_bottom_x(sample_ids, cutoff)
            top_indices = memory_bank.is_in_top_x(sample_ids, cutoff)
            probable_clean_label_indices = torch.logical_or(bottom_indices, top_indices)
        elif take_bottom:
            probable_clean_label_indices = memory_bank.is_in_bottom_x(sample_ids, cutoff)
        else:
            probable_clean_label_indices = memory_bank.is_in_top_x(sample_ids, cutoff)

        return torch.logical_and(probable_clean_label_indices,
                                 incorrect_label_indices).sum().item() / sample_ids.size()[0]

    @staticmethod
    def _get_if_clean_indices(
            memory_banks: List[LossMemoryBank],
            group_ids: torch.Tensor,
            group_membership_ids: torch.Tensor,
            cutoff: int) -> torch.Tensor:
        """
        Returns a selection mask that matches `sample_ids` with true for those elements that are likely to have
        accurate labels, as determined by the `memory_banks`.
        :param memory_banks: Memory memory banks for each IF head measuring sample data, such as loss or entropy.
        :param group_ids The ID of the IF Head responsible for each sample in the mini-batch.
        :param group_membership_ids: The unique ID to be predicted by the IF head for each sample in the mini-batch.
        We use this to subscript into the IF head memory banks.
        :param cutoff: Checks if the samples are in the bottom 'X' number of samples, as measured by the memory bank.
        :return: A selection mask.
        """
        probable_clean_label_indices = torch.zeros_like(group_membership_ids).type(torch.bool)
        for class_index in range(len(memory_banks)):
            current_class_batch_indices = group_ids == class_index
            batch_items_in_class = current_class_batch_indices.sum().item()
            if batch_items_in_class > 0:
                this_class_group_membership_ids = group_membership_ids[current_class_batch_indices]
                bank_ = memory_banks[class_index]
                this_class_probable_clean_label_indices = bank_.is_in_top_x(this_class_group_membership_ids, cutoff)
                probable_clean_label_indices[current_class_batch_indices] = this_class_probable_clean_label_indices
        return probable_clean_label_indices

    def _train_task_asif(self,
                         epoch: int,
                         label_id: torch.Tensor,
                         true_label_id: torch.Tensor,
                         group_ids: torch.Tensor,
                         group_membership_ids: torch.Tensor,
                         sample_ids: torch.Tensor,
                         probably_wrong: torch.Tensor,
                         feature_vectors: torch.Tensor,
                         correct_label_indices: torch.Tensor,
                         incorrect_label_indices: torch.Tensor,
                         config_name: str,
                         trial_index: int,
                         groups_in_batch: List[int],
                         task_losses: Dict[str, List[Dict[str, AverageMeter]]],
                         loss_stats: Dict[str, List[Dict[int, Dict[str, float]]]]) -> torch.Tensor:
        """
        Performs the ASIF task for training. Returns a loss if appropriate. Otherwise, returns None.
        :param epoch: The current training epoch.
        :param label_id: The labels for each sample in the mini-batch.
        :param true_label_id: The true labels for each sample in the mini-batch. Any labels that are withheld or
        incorrect in `label_id` are correctly reflected here.
        :param group_ids The ID of the IF Head responsible for each sample in the mini-batch.
        :param group_membership_ids: The unique ID to be predicted by the IF head for each sample in the mini-batch.
        :param sample_ids: The dataset-wide unique ID for each sample in the mini-batch.
        :param probably_wrong: If 1, the label may not be correct.
        :param feature_vectors: The feature vectors returned from the model.
        :param correct_label_indices: The indices of the samples within the mini-batch that are correctly labelled.
        :param incorrect_label_indices: The indices of the samples within the mini-batch that are incorrectly labelled.
        :param config_name: The name of the configuration being trained.
        :param trial_index: The trial index being trained.
        :param groups_in_batch: List of ASIF groups that were represented in this batch.
        :param task_losses: Accumulates loss statistics.
        :param loss_stats: Collects loss statistics on individual samples.
        :return: The loss resulting from the task. Can be None.
        """

        probably_right = probably_wrong == 0
        if probably_right.sum().item() == 0:
            return None

        if epoch < self.configurations[config_name].asif_start_epoch:
            # Allow a few epochs to establish classification before using ASIF...
            return None

        # Filter out the ones that are probably wrong...
        feature_vectors = feature_vectors[probably_right]
        correct_label_indices = correct_label_indices[probably_right]
        incorrect_label_indices = incorrect_label_indices[probably_right]
        group_ids = group_ids[probably_right]
        group_membership_ids = group_membership_ids[probably_right]
        sample_ids = sample_ids[probably_right]
        true_label_id = true_label_id[probably_right]
        label_id = label_id[probably_right]

        if_logits = []
        for current_group in range(len(self.training_data.samples_per_group)):
            current_class_batch_indices = group_ids == current_group
            batch_items_in_class = current_class_batch_indices.sum().item()
            if batch_items_in_class >= 2:  # Need 2 or more for batch norm
                gpu_device = self.get_gpu_for_if_head(config_name, current_group)
                logits = self.trial_data[config_name][trial_index].if_heads[current_group](
                    feature_vectors[current_class_batch_indices, :].to(gpu_device, non_blocking=True))
            else:
                logits = None
            if_logits.append(logits)

        asif_loss = None
        asif_loss_count = 0

        # Perform identity classification and contribute to overall loss...
        ifc_loss = 0.
        ifc_loss_count = 0
        ifc_clean_loss = 0.
        ifc_clean_loss_count = 0
        ifc_dirty_loss = 0.
        ifc_dirty_loss_count = 0
        for current_class in range(len(self.training_data.samples_per_group)):
            current_class_batch_indices = group_ids == current_class

            # Separate out the part of the batch that is the current class...
            batch_items_in_class = current_class_batch_indices.sum().item()
            total_batch_size = label_id.size()[0]
            class_proportion = batch_items_in_class / total_batch_size

            if if_logits[current_class] is not None:
                groups_in_batch.append(current_class)

                gpu_device = self.get_gpu_for_if_head(config_name, current_class)
                # noinspection PyUnresolvedReferences
                gpu_device_index = gpu_device.index

                current_class_correct_label_indices = correct_label_indices[current_class_batch_indices]
                current_class_incorrect_label_indices = \
                    incorrect_label_indices[current_class_batch_indices]

                target_group_membership_ids = group_membership_ids[current_class_batch_indices].to(gpu_device)
                predicted_group_membership_ids = if_logits[current_class]
                group_id_loss = self.group_criteria[gpu_device_index](predicted_group_membership_ids,
                                                                      target_group_membership_ids)
                group_id_loss = group_id_loss.to(torch.device('cuda:0'))  # Gather on single GPU

                task_losses[config_name][trial_index]['ifc{0}'.format(current_class)].update(
                    group_id_loss.item())

                # Save Identify Feature classification losses...
                ifc_loss += group_id_loss.item()
                ifc_loss_count += 1

                if current_class_correct_label_indices.sum().item() > 0:
                    partial_loss = self.group_criteria[gpu_device_index](
                        predicted_group_membership_ids[current_class_correct_label_indices],
                        target_group_membership_ids[current_class_correct_label_indices])
                    ifc_clean_loss += partial_loss.item()
                    ifc_clean_loss_count += 1
                    task_losses[config_name][trial_index]['ifc{0}_clean'.format(current_class)].update(
                        partial_loss.item())

                if current_class_incorrect_label_indices.sum().item() > 0:
                    partial_loss = self.group_criteria[gpu_device_index](
                        predicted_group_membership_ids[current_class_incorrect_label_indices],
                        target_group_membership_ids[current_class_incorrect_label_indices])
                    ifc_dirty_loss += partial_loss.item()
                    ifc_dirty_loss_count += 1
                    task_losses[config_name][trial_index]['ifc{0}_dirty'.format(current_class)].update(
                        partial_loss.item())

                # Track how many dirty labels were accidentally recorded as 'probably clean'...
                if epoch > self.configurations[config_name].energy_collection_epoch:
                    dirty_selection_percentage = self._calc_dirty_marked_clean_percent(
                        memory_bank=self.trial_data[config_name][trial_index].asif_energy_memory_bank,
                        sample_ids=sample_ids[current_class_batch_indices],
                        incorrect_label_indices=incorrect_label_indices[current_class_batch_indices],
                        cutoff=self.configurations[config_name].true_label_estimate_loss_cutoff)
                    task_losses[config_name][trial_index]['asif_sel_dirty'].update(dirty_selection_percentage)

                    trial_data = self.trial_data[config_name][trial_index]
                    samples_per_if_head = trial_data.if_heads[current_class].out_features
                    for level_ in [20, 40, 60, 80]:
                        dirty_selection_percentage = self._calc_dirty_marked_clean_percent(
                            memory_bank=trial_data.if_loss_memory_banks[current_class],
                            sample_ids=target_group_membership_ids,
                            incorrect_label_indices=incorrect_label_indices[current_class_batch_indices],
                            cutoff=int(samples_per_if_head * level_ / 100))
                        task_losses[config_name][trial_index][f'if_loss_sel_dirty_{level_}'].update(
                            dirty_selection_percentage)
                        dirty_selection_percentage = self._calc_dirty_marked_clean_percent(
                            memory_bank=trial_data.if_entropy_memory_banks[current_class],
                            sample_ids=target_group_membership_ids,
                            incorrect_label_indices=incorrect_label_indices[current_class_batch_indices],
                            cutoff=int(samples_per_if_head * level_ / 100))
                        task_losses[config_name][trial_index][f'if_entropy_sel_dirty_{level_}'].update(
                            dirty_selection_percentage)

                # Calculate per-sample energy and record it for later...
                if epoch >= self.configurations[config_name].energy_collection_epoch:
                    samples_to_measure = if_logits[current_class]
                    samples_apparent_labels = label_id[current_class_batch_indices]
                    samples_true_labels = true_label_id[current_class_batch_indices]
                    samples_ids = sample_ids[current_class_batch_indices]

                    energy_temperature = 1.5
                    per_item_energy = \
                        energy_temperature * torch.logsumexp(samples_to_measure / energy_temperature, dim=1)
                    distributions = functional.softmax(samples_to_measure, dim=1)
                    per_item_entropies = -(torch.log(distributions) * distributions).sum(dim=1)
                    for i in range(distributions.size()[0]):
                        entropy = per_item_entropies[i].item()
                        energy = per_item_energy[i].item()
                        if not math.isnan(energy):
                            task_losses[config_name][trial_index]['ife'].update(energy)
                            task_losses[config_name][trial_index]['ife{0}'.format(current_class)].update(energy)
                        if not math.isnan(entropy):
                            task_losses[config_name][trial_index]['ifs'].update(entropy)
                            task_losses[config_name][trial_index]['ifs{0}'.format(current_class)].update(entropy)
                        if samples_true_labels[i] == samples_apparent_labels[i]:
                            if not math.isnan(energy):
                                task_losses[config_name][trial_index]['ife_clean'].update(energy)
                                task_losses[config_name][trial_index]['ife{0}_clean'.format(current_class)].update(
                                    energy)
                            if not math.isnan(entropy):
                                task_losses[config_name][trial_index]['ifs_clean'].update(entropy)
                                task_losses[config_name][trial_index]['ifs{0}_clean'.format(current_class)].update(
                                    entropy)
                        else:
                            if not math.isnan(energy):
                                task_losses[config_name][trial_index]['ife_dirty'].update(energy)
                                task_losses[config_name][trial_index]['ife{0}_dirty'.format(current_class)].update(
                                    energy)
                            if not math.isnan(entropy):
                                task_losses[config_name][trial_index]['ifs_dirty'].update(entropy)
                                task_losses[config_name][trial_index]['ifs{0}_dirty'.format(current_class)].update(
                                    entropy)

                    if epoch == self.configurations[config_name].energy_collection_epoch:
                        # Only update for the one epoch and then rely on the marked samples thereafter...
                        self.trial_data[config_name][trial_index].if_entropy_memory_banks[current_class].add_to_memory(
                                sample_indices=target_group_membership_ids,
                                labels=samples_apparent_labels,
                                measurement=per_item_entropies,
                                true_labels=samples_true_labels)
                        self.trial_data[config_name][trial_index].asif_energy_memory_bank.add_to_memory(
                            sample_indices=samples_ids,
                            labels=samples_apparent_labels,
                            measurement=per_item_energy,
                            true_labels=samples_true_labels)

                    # Collect per-head losses...
                    per_item_losses = torch.zeros_like(samples_apparent_labels).type(torch.float32)
                    for i in range(samples_apparent_labels.size()[0]):
                        per_head_group_id_loss = self.group_criteria[gpu_device_index](
                            predicted_group_membership_ids[i, :].unsqueeze(0),
                            target_group_membership_ids[i].unsqueeze(0))
                        per_item_losses[i] = per_head_group_id_loss.to(torch.device('cuda:0'))  # Gather on single GPU
                    self.trial_data[config_name][trial_index].if_loss_memory_banks[current_class].add_to_memory(
                            sample_indices=target_group_membership_ids,
                            labels=samples_apparent_labels,
                            measurement=per_item_losses,
                            true_labels=samples_true_labels)

                # Apply the loss in proportion to the amount of the batch that contains this
                # class...
                asif_loss_count += 1
                if asif_loss is None:
                    asif_loss = group_id_loss * class_proportion
                else:
                    asif_loss += group_id_loss * class_proportion

                # Update the lambda value based on the recent group loss...
                self.trial_data[config_name][trial_index].if_heads[current_class].update_lambda(group_id_loss.item(),
                                                                                                self.current_iteration)

        if ifc_loss_count > 0:
            task_losses[config_name][trial_index]['ifc'].update(ifc_loss / ifc_loss_count)
        if ifc_clean_loss_count > 0:
            task_losses[config_name][trial_index]['ifc_clean'].update(ifc_clean_loss / ifc_clean_loss_count)
        if ifc_dirty_loss_count > 0:
            task_losses[config_name][trial_index]['ifc_dirty'].update(ifc_dirty_loss / ifc_dirty_loss_count)

        if self.output_charts:
            per_item_losses = torch.zeros_like(label_id).type(torch.float32)
            for i in range(sample_ids.size()[0]):
                gpu_device = self.get_gpu_for_if_head(config_name, label_id[i])
                # noinspection PyUnresolvedReferences
                gpu_device_index = gpu_device.index

                sample_id = sample_ids[i].item()
                alt_loss_value = self.group_criteria[gpu_device_index](if_logits[label_id[i]][i, :].unsqueeze(0),
                                                                       group_ids[i].unsqueeze(0)).item()
                loss_stats[config_name][trial_index][sample_id] = \
                    {
                        'if_loss': alt_loss_value,
                        'label_id': label_id[i].item(),
                        'label_wrong': label_id[i].item() != true_label_id[i].item(),
                    }
                per_item_losses[i] = alt_loss_value

            self.trial_data[config_name][trial_index].loss_memory_bank_asif.add_to_memory(
                sample_indices=sample_ids,
                labels=label_id,
                measurement=per_item_losses,
                true_labels=true_label_id)

        if asif_loss is not None:
            asif_loss_coefficient = self.configurations[config_name].asif_loss_coefficient

            # During picking phase, we use the ASIF loss coefficient as a constant.
            # Once we begin training, we ramp up the coefficient over the course of the training...
            if self.is_bad_label_picking_phase_over(epoch, config_name):
                start_iteration = self.first_training_epoch(config_name) * len(self.training_data)
                total_iterations = self.total_iterations - start_iteration
                current_iteration = self.current_iteration - start_iteration
                asif_loss_coefficient *= (current_iteration / total_iterations)

            asif_loss = asif_loss * asif_loss_coefficient / asif_loss_count
        return asif_loss

    def _pick_probable_clean_indices(self,
                                     epoch: int,
                                     config_name: str,
                                     trial_index: int,
                                     sample_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns a selection mask for a batch corresponding to the samples that probably have clean labels.
        :param epoch: The current training epoch.
        :param config_name: The current configuration.
        :param trial_index: The current trial.
        :param sample_ids: The unique IDs for each sample in the batch.
        :return: A selection mask.
        """
        config = self.configurations[config_name]
        if config.class_pick_clean_indices == CleanSamplePickingOption.CLASS_LOSS and \
                epoch > config.loss_collection_epoch:
            # Determine which elements of the batch have a loss low enough to be considered as probably
            # having clean labels...
            return self.trial_data[config_name][trial_index].loss_memory_bank_classification.is_in_bottom_x(
                    sample_ids, config.true_label_estimate_loss_cutoff)
        else:
            return torch.ones_like(sample_ids).type(torch.bool)

    def first_training_epoch(self, config_name: str) -> int:
        """
        Returns the first epoch for actual training. All previous epochs to this one are for label picking.
        :param config_name: The configuration in question.
        """
        config_data = self.configurations[config_name]
        return config_data.bad_label_picking_frequency * (config_data.bad_label_picking_cycle_count - 1)

    def is_bad_label_picking_phase_over(self, epoch: int, config_name: str) -> bool:
        return epoch >= self.first_training_epoch(config_name)

    def _train_task_classification(self,
                                   epoch: int,
                                   label_id: torch.Tensor,
                                   true_label_id: torch.Tensor,
                                   sample_ids: torch.Tensor,
                                   probably_wrong: torch.Tensor,
                                   class_logits: torch.Tensor,
                                   labelled_indices: torch.Tensor,
                                   correct_label_indices: torch.Tensor,
                                   incorrect_label_indices: torch.Tensor,
                                   config_name: str,
                                   trial_index: int,
                                   task_losses: Dict[str, List[Dict[str, AverageMeter]]]) -> torch.Tensor:
        """
        Performs the classification task for training. Returns a loss if appropriate. Otherwise, returns None.
        :param epoch: The current training epoch.
        :param label_id: The labels for each sample in the mini-batch.
        :param true_label_id: The true labels for each sample in the mini-batch. Any labels that are withheld or
        incorrect in `label_id` are correctly reflected here.
        :param sample_ids: The dataset-wide unique ID for each sample in the mini-batch.
        :param probably_wrong: If 1, the label may not be correct.
        :param class_logits: The classification outputs from the model.
        :param labelled_indices: The indices of the samples within the mini-batch that have labels.
        :param correct_label_indices: The indices of the samples within the mini-batch that are correctly labelled.
        :param incorrect_label_indices: The indices of the samples within the mini-batch that are incorrectly labelled.
        :param config_name: The name of the configuration being trained.
        :param trial_index: The trial index being trained.
        :param task_losses: Accumulates loss statistics.
        :return: The loss resulting from the task. Can be None.
        """
        probably_right = probably_wrong == 0
        probably_right_and_labelled = torch.logical_and(probably_right, labelled_indices)

        if probably_right_and_labelled.sum().item() == 0:
            # No labelled data, skip...
            return None

        # Filter to those that have labels and are not probably wrong...
        class_logits_labelled = class_logits[probably_right_and_labelled]
        label_id_labelled = label_id[probably_right_and_labelled]
        true_label_id_labelled = true_label_id[probably_right_and_labelled]
        sample_ids_labelled = sample_ids[probably_right_and_labelled]
        correct_label_indices_labelled = correct_label_indices[probably_right_and_labelled]
        incorrect_label_indices_labelled = incorrect_label_indices[probably_right_and_labelled]

        probable_clean_label_indices = self._pick_probable_clean_indices(
            epoch=epoch,
            config_name=config_name,
            trial_index=trial_index,
            sample_ids=sample_ids_labelled)

        # Calculate the classification loss...
        if probable_clean_label_indices.sum().item() > 0:
            if self.configurations[config_name].use_ce:
                classification_loss = \
                    self.trial_data[config_name][trial_index].classification_criteria(
                        class_logits_labelled[probable_clean_label_indices],
                        label_id_labelled[probable_clean_label_indices])
            elif self.configurations[config_name].use_gce:
                classification_loss = \
                    self.trial_data[config_name][trial_index].gce_criteria(
                        class_logits_labelled[probable_clean_label_indices],
                        label_id_labelled[probable_clean_label_indices],
                        sample_ids_labelled[probable_clean_label_indices])
            elif self.configurations[config_name].use_phuber:
                classification_loss = \
                    self.trial_data[config_name][trial_index].phuber_criteria(
                        class_logits_labelled[probable_clean_label_indices],
                        label_id_labelled[probable_clean_label_indices])
            else:
                raise RuntimeError("use_ce, use_gce or use_phuber must be true.")

            task_losses[config_name][trial_index]['cfc'].update(classification_loss.item())
        else:
            classification_loss = None

        # Save the losses for later attempts to determine which labels are false...
        if epoch >= self.configurations[config_name].loss_collection_epoch:
            per_item_losses = torch.zeros_like(label_id_labelled).type(torch.float32)
            for i in range(sample_ids_labelled.size()[0]):
                alt_loss_value = self.trial_data[config_name][trial_index].classification_criteria(
                    class_logits_labelled[i, :].unsqueeze(0),
                    label_id_labelled[i].unsqueeze(0))
                per_item_losses[i] = alt_loss_value

            self.trial_data[config_name][trial_index].loss_memory_bank_classification.add_to_memory(
                sample_indices=sample_ids_labelled,
                labels=label_id_labelled,
                measurement=per_item_losses,
                true_labels=true_label_id_labelled)

        # Track how many dirty labels were accidentally recorded as 'probably clean'...
        percent_dirty_labels_marked_as_probably_clean = \
            torch.logical_and(
                probable_clean_label_indices,
                incorrect_label_indices_labelled).sum().item() / label_id_labelled.size()[0]
        task_losses[config_name][trial_index]['sel_dirty'].update(percent_dirty_labels_marked_as_probably_clean)

        if self.output_charts:
            # Track how many dirty labels were accidentally recorded as 'probably clean' by ASIF...
            probable_clean_label_indices_asif_10 = \
                self.trial_data[config_name][trial_index].loss_memory_bank_asif.is_in_bottom_x(
                    sample_ids_labelled, 10)
            percent_dirty_labels_marked_as_probably_clean = \
                torch.logical_and(
                    probable_clean_label_indices_asif_10,
                    incorrect_label_indices_labelled).sum().item() / label_id_labelled.size()[0]
            task_losses[config_name][trial_index]['sel_dirty_asif_10'].update(
                percent_dirty_labels_marked_as_probably_clean)

            probable_clean_label_indices_asif_100 = \
                self.trial_data[config_name][trial_index].loss_memory_bank_asif.is_in_bottom_x(
                    sample_ids_labelled, 100)
            percent_dirty_labels_marked_as_probably_clean = \
                torch.logical_and(
                    probable_clean_label_indices_asif_100,
                    incorrect_label_indices_labelled).sum().item() / label_id_labelled.size()[0]
            task_losses[config_name][trial_index]['sel_dirty_asif_100'].update(
                percent_dirty_labels_marked_as_probably_clean)

            probable_clean_label_indices_asif_1000 = \
                self.trial_data[config_name][trial_index].loss_memory_bank_asif.is_in_bottom_x(
                    sample_ids_labelled, 1000)
            percent_dirty_labels_marked_as_probably_clean = \
                torch.logical_and(
                    probable_clean_label_indices_asif_1000,
                    incorrect_label_indices_labelled).sum().item() / label_id_labelled.size()[0]
            task_losses[config_name][trial_index]['sel_dirty_asif_1000'].update(
                percent_dirty_labels_marked_as_probably_clean)

            probable_clean_label_indices_asif_1000_combined = \
                torch.logical_and(probable_clean_label_indices_asif_1000,
                                  probable_clean_label_indices)
            percent_dirty_labels_marked_as_probably_clean = \
                torch.logical_and(
                    probable_clean_label_indices_asif_1000_combined,
                    incorrect_label_indices_labelled).sum().item() / label_id_labelled.size()[0]
            task_losses[config_name][trial_index]['sel_dirty_combined'].update(
                percent_dirty_labels_marked_as_probably_clean)

            task_losses[config_name][trial_index]['sel_dirty_combined_count'].update(
                probable_clean_label_indices_asif_1000_combined.sum().item())

            # Track how many dirty labels were accidentally recorded as 'probably clean' by ASIF...
            probable_clean_label_indices_asif_10 = \
                self.trial_data[config_name][trial_index].loss_memory_bank_asif.is_in_top_x(
                    sample_ids_labelled, 10)
            percent_dirty_labels_marked_as_probably_clean = \
                torch.logical_and(
                    probable_clean_label_indices_asif_10,
                    incorrect_label_indices_labelled).sum().item() / label_id_labelled.size()[0]
            task_losses[config_name][trial_index]['sel_dirty_asif_rev_10'].update(
                percent_dirty_labels_marked_as_probably_clean)

            probable_clean_label_indices_asif_100 = \
                self.trial_data[config_name][trial_index].loss_memory_bank_asif.is_in_top_x(
                    sample_ids_labelled, 100)
            percent_dirty_labels_marked_as_probably_clean = \
                torch.logical_and(
                    probable_clean_label_indices_asif_100,
                    incorrect_label_indices_labelled).sum().item() / label_id_labelled.size()[0]
            task_losses[config_name][trial_index]['sel_dirty_asif_rev_100'].update(
                percent_dirty_labels_marked_as_probably_clean)

            probable_clean_label_indices_asif_1000 = \
                self.trial_data[config_name][trial_index].loss_memory_bank_asif.is_in_top_x(
                    sample_ids_labelled, 1000)
            percent_dirty_labels_marked_as_probably_clean = \
                torch.logical_and(
                    probable_clean_label_indices_asif_1000,
                    incorrect_label_indices_labelled).sum().item() / label_id_labelled.size()[0]
            task_losses[config_name][trial_index]['sel_dirty_asif_rev_1000'].update(
                percent_dirty_labels_marked_as_probably_clean)

            probable_clean_label_indices_asif_1000_combined = \
                torch.logical_and(probable_clean_label_indices_asif_1000,
                                  probable_clean_label_indices)
            percent_dirty_labels_marked_as_probably_clean = \
                torch.logical_and(
                    probable_clean_label_indices_asif_1000_combined,
                    incorrect_label_indices_labelled).sum().item() / label_id_labelled.size()[0]
            task_losses[config_name][trial_index]['sel_dirty_rev_combined'].update(
                percent_dirty_labels_marked_as_probably_clean)

            task_losses[config_name][trial_index]['sel_dirty_rev_combined_count'].update(
                probable_clean_label_indices_asif_1000_combined.sum().item())

        if correct_label_indices_labelled.sum().item() > 0:
            partial_loss = self.trial_data[config_name][trial_index].classification_criteria(
                class_logits_labelled[correct_label_indices_labelled],
                label_id_labelled[correct_label_indices_labelled])
            task_losses[config_name][trial_index]['cfc_clean_lbl'].update(partial_loss.item())

        if incorrect_label_indices_labelled.sum().item() > 0:
            partial_loss = self.trial_data[config_name][trial_index].classification_criteria(
                class_logits_labelled[incorrect_label_indices_labelled],
                label_id_labelled[incorrect_label_indices_labelled])
            task_losses[config_name][trial_index]['cfc_noisy_lbl'].update(partial_loss.item())

        return classification_loss

    def _train_iter(self,
                    epoch,
                    sample: torch.Tensor,
                    label_id: torch.Tensor,
                    true_label_id: torch.Tensor,
                    group_ids: torch.Tensor,
                    group_membership_ids: torch.Tensor,
                    sample_ids: torch.Tensor,
                    probably_wrong: torch.Tensor,
                    task_losses: Dict[str, List[Dict[str, AverageMeter]]],
                    loss_stats: Dict[str, List[Dict[int, Dict[str, float]]]],
                    if_heads_only: bool):
        sample = sample.type(torch.float32)
        label_id = label_id.type(torch.long)
        true_label_id = true_label_id.type(torch.long)
        group_ids = group_ids.type(torch.long)
        group_membership_ids = group_membership_ids.type(torch.long)
        sample_ids = sample_ids.type(torch.long)
        probably_wrong = probably_wrong.type(torch.long)

        if self.use_cuda:
            sample = sample.cuda(non_blocking=True)
            label_id = label_id.cuda(non_blocking=True)
            true_label_id = true_label_id.cuda(non_blocking=True)
            group_ids = group_ids.cuda(non_blocking=True)
            group_membership_ids = group_membership_ids.cuda(non_blocking=True)
            sample_ids = sample_ids.cuda(non_blocking=True)
            probably_wrong = probably_wrong.cuda(non_blocking=True)

        labelled_indices = label_id != -1
        correct_label_indices = label_id == true_label_id
        incorrect_label_indices = label_id != true_label_id

        asif_groups_in_batch = []

        for _config in self.configurations:
            if self.current_iteration >= self.configurations[_config].total_iterations:
                # This configuration is done training, move onto the next one...
                continue

            # Work out which tasks we want to do for training...
            tasks = []
            if not if_heads_only:
                tasks.append('CLASSIFICATION')
            if self.configurations[_config].use_asif:
                tasks.append('ASIF')
            if self.configurations[_config].use_cc and not if_heads_only:
                tasks.append('CC')

            for _trial in range(self.num_trials):
                class_logits, feature_vectors, if_head_inputs = self.trial_data[_config][_trial].model(sample)

                # Start by recording the model's accuracy...
                if labelled_indices.sum().item() > 0:
                    task_losses[_config][_trial]['apparent_accuracy'].update(
                        (class_logits[labelled_indices].argmax(dim=1) == label_id[labelled_indices]).sum().item() * 100.
                        / class_logits[labelled_indices].size()[0])
                task_losses[_config][_trial]['true_accuracy'].update(
                    (class_logits.argmax(dim=1) == true_label_id).sum().item() * 100. / class_logits.size()[0])

                loss = None
                for _task in tasks:
                    if _task == "CLASSIFICATION":
                        classification_loss = self._train_task_classification(
                           epoch,
                           label_id,
                           true_label_id,
                           sample_ids,
                           probably_wrong,
                           class_logits,
                           labelled_indices,
                           correct_label_indices,
                           incorrect_label_indices,
                           _config,
                           _trial,
                           task_losses)

                        if classification_loss is not None:
                            if loss is None:
                                loss = classification_loss
                            else:
                                loss += classification_loss

                    elif _task == "ASIF":
                        asif_loss = self._train_task_asif(
                            epoch,
                            label_id,
                            true_label_id,
                            group_ids,
                            group_membership_ids,
                            sample_ids,
                            probably_wrong,
                            if_head_inputs,
                            correct_label_indices,
                            incorrect_label_indices,
                            _config,
                            _trial,
                            asif_groups_in_batch,
                            task_losses,
                            loss_stats)

                        gc.collect()
                        torch.cuda.empty_cache()

                        if asif_loss is not None:
                            if loss is None:
                                loss = asif_loss
                            else:
                                loss += asif_loss

                    elif _task == "CC":
                        cc_loss = self._train_task_cc(
                            epoch,
                            label_id,
                            probably_wrong,
                            feature_vectors,
                            labelled_indices,
                            _config,
                            _trial,
                            task_losses)

                        if cc_loss is not None:
                            if loss is None:
                                loss = cc_loss
                            else:
                                loss += cc_loss

                if loss is not None:
                    if_heads_to_train = []
                    for i in asif_groups_in_batch:
                        if i not in if_heads_to_train:
                            if_heads_to_train.append(i)

                    self.trial_data[_config][_trial].optimizers['model_optimizer'].zero_grad(set_to_none=True)
                    if self.configurations[_config].use_asif:
                        for i in if_heads_to_train:
                            self.trial_data[_config][_trial].optimizers['if_feature_optimizer{0}'.format(i)].zero_grad(
                                set_to_none=True)
                    loss.backward()
                    self.trial_data[_config][_trial].optimizers['model_optimizer'].step()
                    if self.configurations[_config].use_asif:
                        for i in if_heads_to_train:
                            self.trial_data[_config][_trial].optimizers['if_feature_optimizer{0}'.format(i)].step()

                        # Recreate the optimizers.
                        # The optimizers hold on to a lot of CUDA memory that they don't need to hold onto. When we have
                        # 100+ optimizers, we can't afford the waste. If any readers know how to clean up without re-
                        # creating the optimizers, please let me know...
                        optimizer_keys = [key for key in self.trial_data[_config][_trial].optimizers]
                        for key in optimizer_keys:
                            del self.trial_data[_config][_trial].optimizers[key]
                        self.trial_data[_config][_trial].create_optimizers(epoch=epoch, force_creation=True)

                    del loss

    def _record_cls_loss(self,
                         epoch: int,
                         config_name: str,
                         trial_index: int,
                         classification_logits: torch.Tensor,
                         label_id: torch.Tensor,
                         true_label_id: torch.Tensor,
                         sample_indices: torch.Tensor,
                         task_losses: Dict[str, List[Dict[str, AverageMeter]]]):
        classification_per_item_losses = \
            self.trial_data[config_name][trial_index].classification_per_sample_criteria(classification_logits,
                                                                                         label_id)
        classification_losses_mean = classification_per_item_losses.mean(dim=0).item()
        classification_losses_stdev = classification_per_item_losses.std(dim=0).item()
        classification_per_item_losses_normalised = \
            (classification_per_item_losses - classification_losses_mean) / classification_losses_stdev

        self.trial_data[config_name][trial_index].cls_memory_bank.add_to_memory(
            epoch=epoch,
            sample_indices=sample_indices,
            labels=label_id,
            measurement=classification_per_item_losses,
            true_labels=true_label_id)

        for i in range(classification_per_item_losses_normalised.size()[0]):
            task_losses[config_name][trial_index]['cls'].update(classification_per_item_losses_normalised[i].item())
            if true_label_id[i] == label_id[i]:
                task_losses[config_name][trial_index]['cls_clean'].update(
                    classification_per_item_losses_normalised[i].item())
            else:
                task_losses[config_name][trial_index]['cls_dirty'].update(
                    classification_per_item_losses_normalised[i].item())

    def _record_cls_entropy(self,
                            epoch: int,
                            config_name: str,
                            trial_index: int,
                            classification_logits: torch.Tensor,
                            label_id: torch.Tensor,
                            true_label_id: torch.Tensor,
                            sample_indices: torch.Tensor,
                            task_losses: Dict[str, List[Dict[str, AverageMeter]]]):

        distributions = functional.softmax((classification_logits - classification_logits.mean(dim=0)) /
                                           classification_logits.std(dim=0), dim=1)
        per_item_entropies = -(torch.log(distributions) * distributions).sum(dim=1)

        entropy_mean = per_item_entropies.mean(dim=0).item()
        entropy_stdev = per_item_entropies.std(dim=0).item()
        per_item_entropies_normalised = (per_item_entropies - entropy_mean) / entropy_stdev

        self.trial_data[config_name][trial_index].cls_entropy_memory_bank.add_to_memory(
            epoch=epoch,
            sample_indices=sample_indices,
            labels=label_id,
            measurement=per_item_entropies_normalised,
            true_labels=true_label_id)

        for i in range(per_item_entropies_normalised.size()[0]):
            task_losses[config_name][trial_index]['clss'].update(per_item_entropies_normalised[i].item())
            if true_label_id[i] == label_id[i]:
                task_losses[config_name][trial_index]['clss_clean'].update(per_item_entropies_normalised[i].item())
            else:
                task_losses[config_name][trial_index]['clss_dirty'].update(per_item_entropies_normalised[i].item())

    def _record_cross_head_relative_entropy(self,
                                            epoch: int,
                                            config_name: str,
                                            trial_index: int,
                                            if_logits: List[torch.Tensor],
                                            label_id: torch.Tensor,
                                            true_label_id: torch.Tensor,
                                            group_ids: torch.Tensor,
                                            group_membership_ids: torch.Tensor,
                                            sample_indices: torch.Tensor,
                                            task_losses: Dict[str, List[Dict[str, AverageMeter]]]):
        # Get the logits from each of the IF heads...
        per_group_per_item_entropies = []
        for current_group in range(len(self.training_data.samples_per_group)):
            logits = if_logits[current_group]

            # Work out loss for those samples that belong to this head...
            current_class_batch_indices = group_ids == current_group
            batch_items_in_class = current_class_batch_indices.sum().item()
            if batch_items_in_class > 0:
                distributions = functional.softmax((logits - logits.mean(dim=0)) / logits.std(dim=0), dim=1)
                per_item_entropies = -(torch.log(distributions) * distributions).sum(dim=1)

                group_id_loss = self.asif_per_sample_criteria(logits[current_class_batch_indices],
                                                              group_membership_ids[current_class_batch_indices])

                self.trial_data[config_name][trial_index].asif_loss_memory_bank.add_to_memory(
                    epoch=epoch,
                    sample_indices=sample_indices[current_class_batch_indices],
                    labels=label_id[current_class_batch_indices],
                    measurement=group_id_loss,
                    true_labels=true_label_id[current_class_batch_indices])

                self.trial_data[config_name][trial_index].entropy_head_memory_bank.add_to_memory(
                    epoch=epoch,
                    sample_indices=sample_indices[current_class_batch_indices],
                    labels=label_id[current_class_batch_indices],
                    measurement=per_item_entropies[current_class_batch_indices],
                    true_labels=true_label_id[current_class_batch_indices])

            # Work out entropy for each IF head. This includes for samples that are not meant for a given IF head...
            distributions = functional.softmax((logits - logits.mean(dim=0)) / logits.std(dim=0), dim=1)
            per_item_entropies = -(torch.log(distributions) * distributions).sum(dim=1)
            for i in range(distributions.size()[0]):
                entropy = per_item_entropies[i].item()
                if not math.isnan(entropy):
                    task_losses[config_name][trial_index]['ifs'].update(entropy)
                if true_label_id[i] == label_id[i]:
                    if not math.isnan(entropy):
                        task_losses[config_name][trial_index]['ifs_clean'].update(entropy)
                else:
                    if not math.isnan(entropy):
                        task_losses[config_name][trial_index]['ifs_dirty'].update(entropy)

            per_group_per_item_entropies.append(per_item_entropies)

        # For each sample, calculate the number of standard deviations between the entropy behind the correct
        # IF head and the average for all IF heads.
        stacked_entropies = torch.stack(per_group_per_item_entropies)
        _, entropy_indices = stacked_entropies.sort(dim=0)
        entropies_mean = stacked_entropies.mean(dim=0)
        entropies_stdev = stacked_entropies.std(dim=0)
        entropies_min, entropies_min_indices = stacked_entropies.min(dim=0)
        entropies_max, entropies_max_indices = stacked_entropies.max(dim=0)
        correct_entropies = torch.zeros_like(entropies_mean)  # The entropy from the correct IF heads
        apparent_entropies = torch.zeros_like(entropies_mean)  # The entropy from the apparent IF heads
        correct_entropy_indices = torch.zeros_like(entropies_mean)  # The entropy indices from the correct IF heads
        apparent_entropy_indices = torch.zeros_like(entropies_mean)  # The entropy indices from the apparent heads
        for i in range(correct_entropies.size()[0]):
            # Work out the correct IF head...
            correct_head = true_label_id[i].item()
            correct_head_entropy = per_group_per_item_entropies[correct_head][i].item()
            correct_entropies[i] = correct_head_entropy

            # Work out the apparent correct IF head...
            apparent_head = label_id[i].item()
            apparent_head_head_entropy = per_group_per_item_entropies[apparent_head][i].item()
            apparent_entropies[i] = apparent_head_head_entropy

            # Work out place in the order of the entropy from the heads. e.g. If the head's entropy is the
            # lowest, it should be '0'. If it is highest, it should be '9'...
            correct_entropy_indices[i] = entropy_indices[correct_head][i]
            apparent_entropy_indices[i] = entropy_indices[apparent_head][i]

        correct_entropies_normalised = (correct_entropies - entropies_mean) / entropies_stdev
        apparent_entropies_normalised = (apparent_entropies - entropies_mean) / entropies_stdev
        min_entropies_normalised = (entropies_min - entropies_mean) / entropies_stdev
        max_entropies_normalised = (entropies_max - entropies_mean) / entropies_stdev

        self.trial_data[config_name][trial_index].entropy_head_samplewise_memory_bank.add_to_memory(
            epoch=epoch,
            sample_indices=sample_indices,
            labels=label_id,
            measurement=apparent_entropies_normalised,
            true_labels=true_label_id)

        for i in range(correct_entropies.size()[0]):
            if not math.isnan(apparent_entropies_normalised[i].item()):
                task_losses[config_name][trial_index]['ifsan'].update(apparent_entropies_normalised[i].item())
                if true_label_id[i] == label_id[i]:
                    task_losses[config_name][trial_index]['ifsan_clean'].update(apparent_entropies_normalised[i].item())
                else:
                    task_losses[config_name][trial_index]['ifsan_dirty'].update(apparent_entropies_normalised[i].item())

                task_losses[config_name][trial_index]['ifscn'].update(correct_entropies_normalised[i].item())
                task_losses[config_name][trial_index]['ifsminn'].update(min_entropies_normalised[i].item())
                task_losses[config_name][trial_index]['ifsmaxn'].update(max_entropies_normalised[i].item())

                task_losses[config_name][trial_index]['ifsi'].update(apparent_entropy_indices[i].item())
                if true_label_id[i] == label_id[i]:
                    task_losses[config_name][trial_index]['ifsi_clean'].update(apparent_entropy_indices[i].item())
                else:
                    task_losses[config_name][trial_index]['ifsi_dirty'].update(apparent_entropy_indices[i].item())

    def _record_cross_head_relative_energy(self,
                                           epoch: int,
                                           config_name: str,
                                           trial_index: int,
                                           if_logits: List[torch.Tensor],
                                           label_id: torch.Tensor,
                                           true_label_id: torch.Tensor,
                                           group_ids: torch.Tensor,
                                           group_membership_ids: torch.Tensor,
                                           sample_indices: torch.Tensor,
                                           task_losses: Dict[str, List[Dict[str, AverageMeter]]]):
        # Get the logits from each of the IF heads...
        per_group_per_item_energy = []
        for current_group in range(len(self.training_data.samples_per_group)):
            logits = if_logits[current_group]

            # Work out loss for those samples that belong to this head...
            current_class_batch_indices = group_ids == current_group
            batch_items_in_class = current_class_batch_indices.sum().item()
            if batch_items_in_class > 0:
                distributions = functional.softmax((logits - logits.mean(dim=0)) / logits.std(dim=0), dim=1)
                per_item_entropies = -(torch.log(distributions) * distributions).sum(dim=1)

                group_id_loss = self.asif_per_sample_criteria(logits[current_class_batch_indices],
                                                              group_membership_ids[current_class_batch_indices])

                self.trial_data[config_name][trial_index].asif_loss_memory_bank.add_to_memory(
                    epoch=epoch,
                    sample_indices=sample_indices[current_class_batch_indices],
                    labels=label_id[current_class_batch_indices],
                    measurement=group_id_loss,
                    true_labels=true_label_id[current_class_batch_indices])

                self.trial_data[config_name][trial_index].entropy_head_memory_bank.add_to_memory(
                    epoch=epoch,
                    sample_indices=sample_indices[current_class_batch_indices],
                    labels=label_id[current_class_batch_indices],
                    measurement=per_item_entropies[current_class_batch_indices],
                    true_labels=true_label_id[current_class_batch_indices])

            energy_temperature = 1.5
            per_item_energy = energy_temperature * torch.logsumexp(logits / energy_temperature, dim=1)
            for i in range(per_item_energy.size()[0]):
                energy = per_item_energy[i].item()
                if not math.isnan(energy):
                    task_losses[config_name][trial_index]['ife'].update(energy)
                if true_label_id[i] == label_id[i]:
                    if not math.isnan(energy):
                        task_losses[config_name][trial_index]['ife_clean'].update(energy)
                else:
                    if not math.isnan(energy):
                        task_losses[config_name][trial_index]['ife_dirty'].update(energy)
            per_group_per_item_energy.append(per_item_energy)

        # For each sample, calculate the number of standard deviations between the energy behind the correct
        # IF head and the average for all IF heads.
        stacked_energies = torch.stack(per_group_per_item_energy)
        _, energy_indices = stacked_energies.sort(dim=0)
        energies_mean = stacked_energies.mean(dim=0)
        energies_stdev = stacked_energies.std(dim=0)
        energies_min, energies_min_indices = stacked_energies.min(dim=0)
        energies_max, energies_max_indices = stacked_energies.max(dim=0)
        correct_energies = torch.zeros_like(energies_mean)  # The energy from the correct IF heads
        apparent_energies = torch.zeros_like(energies_mean)  # The energy from the apparent IF heads
        correct_energies_indices = torch.zeros_like(energies_mean)  # The energy indices from the correct IF heads
        apparent_energies_indices = torch.zeros_like(energies_mean)  # The energy indices from the apparent heads
        for i in range(correct_energies.size()[0]):
            # Work out the correct IF head...
            correct_head = true_label_id[i].item()
            correct_head_energy = per_group_per_item_energy[correct_head][i].item()
            correct_energies[i] = correct_head_energy

            # Work out the apparent correct IF head...
            apparent_head = label_id[i].item()
            apparent_head_energy = per_group_per_item_energy[apparent_head][i].item()
            apparent_energies[i] = apparent_head_energy

            # Work out place in the order of the energy from the heads. e.g. If the head's energy is the
            # lowest, it should be '0'. If it is highest, it should be '9'...
            correct_energies_indices[i] = energy_indices[correct_head][i]
            apparent_energies_indices[i] = energy_indices[apparent_head][i]

        correct_energies_normalised = (correct_energies - energies_mean) / energies_stdev
        apparent_energies_normalised = (apparent_energies - energies_mean) / energies_stdev
        min_energies_normalised = (energies_min - energies_mean) / energies_stdev
        max_energies_normalised = (energies_max - energies_mean) / energies_stdev

        self.trial_data[config_name][trial_index].energy_head_samplewise_memory_bank.add_to_memory(
            epoch=epoch,
            sample_indices=sample_indices,
            labels=label_id,
            measurement=apparent_energies_normalised,
            true_labels=true_label_id)

        for i in range(correct_energies.size()[0]):
            if not math.isnan(apparent_energies_normalised[i].item()):
                task_losses[config_name][trial_index]['ifean'].update(apparent_energies_normalised[i].item())
                if true_label_id[i] == label_id[i]:
                    task_losses[config_name][trial_index]['ifean_clean'].update(apparent_energies_normalised[i].item())
                else:
                    task_losses[config_name][trial_index]['ifean_dirty'].update(apparent_energies_normalised[i].item())

                task_losses[config_name][trial_index]['ifecn'].update(correct_energies_normalised[i].item())
                task_losses[config_name][trial_index]['ifeminn'].update(min_energies_normalised[i].item())
                task_losses[config_name][trial_index]['ifemaxn'].update(max_energies_normalised[i].item())

                task_losses[config_name][trial_index]['ifei'].update(apparent_energies_indices[i].item())
                if true_label_id[i] == label_id[i]:
                    task_losses[config_name][trial_index]['ifei_clean'].update(apparent_energies_indices[i].item())
                else:
                    task_losses[config_name][trial_index]['ifei_dirty'].update(apparent_energies_indices[i].item())

    def _eval_if_heads_iter(self,
                            epoch: int,
                            sample: torch.Tensor,
                            label_id: torch.Tensor,
                            true_label_id: torch.Tensor,
                            group_ids: torch.Tensor,
                            group_membership_ids: torch.Tensor,
                            sample_indices: torch.Tensor,
                            task_losses: Dict[str, List[Dict[str, AverageMeter]]]):

        sample = sample.type(torch.float32)
        label_id = label_id.type(torch.long)
        true_label_id = true_label_id.type(torch.long)
        group_ids = group_ids.type(torch.long)
        group_membership_ids = group_membership_ids.type(torch.long)
        sample_indices = sample_indices.type(torch.long)

        if self.use_cuda:
            sample = sample.cuda(non_blocking=True)
            label_id = label_id.cuda(non_blocking=True)
            true_label_id = true_label_id.cuda(non_blocking=True)
            group_ids = group_ids.cuda(non_blocking=True)
            group_membership_ids = group_membership_ids.cuda(non_blocking=True)
            sample_indices = sample_indices.cuda(non_blocking=True)

        for _config in self.trial_data:
            for _trial in range(self.num_trials):
                class_logits, feature_vectors, if_head_inputs = self.trial_data[_config][_trial].model(sample)

                # Start by recording the model's accuracy...
                task_losses[_config][_trial]['apparent_accuracy'].update(
                    (class_logits.argmax(dim=1) == label_id).sum().item() * 100.
                    / class_logits.size()[0])
                task_losses[_config][_trial]['true_accuracy'].update(
                    (class_logits.argmax(dim=1) == true_label_id).sum().item() * 100. / class_logits.size()[0])

                self._record_cls_loss(epoch,
                                      _config,
                                      _trial,
                                      class_logits,
                                      label_id,
                                      true_label_id,
                                      sample_indices,
                                      task_losses)
                self._record_cls_entropy(epoch,
                                         _config,
                                         _trial,
                                         class_logits,
                                         label_id,
                                         true_label_id,
                                         sample_indices,
                                         task_losses)

                # Get output from IF heads...
                if self.configurations[_config].use_asif:
                    if_logits = []
                    for current_group in range(len(self.training_data.samples_per_group)):
                        if_logits.append(self.trial_data[_config][_trial].if_heads[current_group](if_head_inputs))
                    self._record_cross_head_relative_entropy(epoch,
                                                             _config,
                                                             _trial,
                                                             if_logits,
                                                             label_id,
                                                             true_label_id,
                                                             group_ids,
                                                             group_membership_ids,
                                                             sample_indices,
                                                             task_losses)
                    self._record_cross_head_relative_energy(epoch,
                                                            _config,
                                                            _trial,
                                                            if_logits,
                                                            label_id,
                                                            true_label_id,
                                                            group_ids,
                                                            group_membership_ids,
                                                            sample_indices,
                                                            task_losses)

    def _eval_iter(self,
                   sample: torch.Tensor,
                   classification_id: torch.Tensor,
                   stats: Dict[str, List[PerformanceData]],
                   accuracies: Dict[str, List[Accuracy]]):

        sample = sample.type(torch.float32)
        classification_id = classification_id.type(torch.long)

        if self.use_cuda:
            sample = sample.cuda(non_blocking=True)
            classification_id = classification_id.cuda(non_blocking=True)

        for _config in self.trial_data:
            for _trial in range(self.num_trials):
                self._eval_iter_trial(sample, classification_id, stats, accuracies, _config, _trial)

    def _eval_iter_trial(self,
                         sample: torch.Tensor,
                         classification_id: torch.Tensor,
                         stats: Dict[str, List[PerformanceData]],
                         accuracies: Dict[str, List[Accuracy]],
                         config_name: str,
                         trial_index: int):
        label_pred, _, _ = self.trial_data[config_name][trial_index].model(sample)
        label_pred = label_pred.cpu().detach()

        for _item in range(label_pred.size()[0]):
            # Calculate accuracy using regular classifier...
            accuracies[config_name][trial_index].update_state(classification_id[_item].item(),
                                                              torch.argmax(label_pred[_item]).item())

            for field_id in range(self.testing_data.class_count):
                prediction = torch.argmax(label_pred[_item]).item() == field_id
                truth = classification_id[_item].item() == field_id

                stats[config_name][trial_index].add_observation(field_id, prediction, truth)

    def _should_keep_training(self) -> bool:
        return self.current_iteration < self.total_iterations

    def create_classification_criteria(self, reduction: str, config_name: str) -> \
            Union[nn.CrossEntropyLoss, nn.NLLLoss]:
        if self.configurations[config_name].loss == LossOption.LOSS_CE:
            class_weights = torch.zeros(self.training_data.class_count)
            for i in range(self.training_data.class_count):
                class_weights[i] = 1 / self.training_data.samples_per_label[i]
            return nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
        elif self.configurations[config_name].loss == LossOption.LOSS_NLL:
            # Do not use weights since IDN did not. We may revisit this decision...
            return nn.NLLLoss(reduction=reduction)
        else:
            raise ValueError(f"'{self.configurations[config_name].loss.get_short_name()}' is not a supported loss.")

    def _create_group_criteria(self) -> List[nn.CrossEntropyLoss]:
        if self.split_across_gpus:
            criteria = []
            for _ in range(torch.cuda.device_count()):
                criteria.append(nn.CrossEntropyLoss())
            return criteria
        else:
            return [nn.CrossEntropyLoss()]

    @staticmethod
    def _calc_loss_stats(loss_data: Dict[int, Dict[str, float]]) -> List[Tuple[float, float, float, float]]:
        """
        Calculates the loss stats and returns them.
        :argument loss_data Dictionary of loss data.
        :returns A list of tuple of four values:
            - Mean loss on correct labels for a given label.
            - Standard Deviation of loss on correct labels.
            - Mean loss on incorrect labels for a given label.
            - Standard Deviation of loss on incorrect labels.
        """
        to_return = []
        for label_id in range(10):
            data = []
            for index_ in loss_data:
                record_ = loss_data[index_]
                if record_['label_id'] == label_id:
                    data.append((record_['if_loss'], record_['label_wrong']))
            wrong_losses = [data[i][0] for i in range(len(data)) if data[i][1]]
            if len(wrong_losses) > 0:
                wrong_mean = statistics.mean(wrong_losses)
                if len(wrong_losses) > 2:
                    wrong_stdev = statistics.stdev(wrong_losses)
                else:
                    wrong_stdev = 0.
            else:
                wrong_mean = 0.
                wrong_stdev = 0.

            right_losses = [data[i][0] for i in range(len(data)) if not data[i][1]]
            if len(right_losses) > 0:
                right_mean = statistics.mean(right_losses)
                if len(right_losses) > 2:
                    right_stdev = statistics.stdev(right_losses)
                else:
                    right_stdev = 0.
            else:
                right_mean = 0.
                right_stdev = 0.

            to_return.append((right_mean, right_stdev, wrong_mean, wrong_stdev))
        return to_return

    def _write_incorrect_selection_data_chart(self,
                                              epoch: int,
                                              trial_index: int,
                                              config: str):
        if epoch == 0:
            return

        if self.configurations[config].use_asif:
            rows = 2
        else:
            rows = 1

        fig = plt.figure(figsize=(24, 12))
        ax = fig.add_subplot(rows, 1, 1)
        ax.set_title(f'False Pick % vs Selection Count', fontsize=12)

        memory_bank_classification = self.trial_data[config][trial_index].loss_memory_bank_classification
        max_selection = memory_bank_classification.sample_measurement_ordering_ascending.max().item()
        false_labelled = (memory_bank_classification.sample_true_labels != memory_bank_classification.sample_labels)

        selections = np.arange(max_selection)
        noisy_selections_ascending = np.zeros(max_selection)
        noisy_selections_descending = np.zeros(max_selection)
        for selection in range(max_selection):
            is_selected = memory_bank_classification.sample_measurement_ordering_ascending < (selection + 1)
            is_false_and_selected = torch.logical_and(is_selected, false_labelled)
            noisy_selections_ascending[selection] = is_false_and_selected.sum().item() * 100. / is_selected.sum().item()

            is_selected = memory_bank_classification.sample_measurement_ordering_descending < (selection + 1)
            is_false_and_selected = torch.logical_and(is_selected, false_labelled)
            noisy_selections_descending[selection] = \
                is_false_and_selected.sum().item() * 100. / is_selected.sum().item()

        ax.plot(selections, noisy_selections_ascending, label='Ascending Classification Loss')
        ax.plot(selections, noisy_selections_descending, label='Descending Classification Loss')

        # Output the ASIF data...
        if self.configurations[config].use_asif:
            memory_bank_asif = self.trial_data[config][trial_index].loss_memory_bank_asif
            max_selection = memory_bank_asif.sample_measurement_ordering_ascending.max().item()
            false_labelled = (memory_bank_asif.sample_true_labels != memory_bank_asif.sample_labels)

            selections = np.arange(max_selection)
            noisy_selections_ascending = np.zeros(max_selection)
            noisy_selections_descending = np.zeros(max_selection)
            for selection in range(max_selection):
                is_selected = memory_bank_asif.sample_measurement_ordering_ascending < (selection + 1)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                noisy_selections_ascending[selection] = \
                    is_false_and_selected.sum().item() * 100. / is_selected.sum().item()

                is_selected = memory_bank_asif.sample_measurement_ordering_descending < (selection + 1)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                noisy_selections_descending[selection] = \
                    is_false_and_selected.sum().item() * 100. / is_selected.sum().item()

            ax.plot(selections, noisy_selections_ascending, label='Ascending ASIF Loss')
            ax.plot(selections, noisy_selections_descending, label='Descending ASIF Loss')
            ax.legend()

            # Output ascending class vs ascending ASIF combination...
            ax = fig.add_subplot(2, 1, 2)
            ax.set_title(f'Total Selections (Classification and ASIF combined)', fontsize=12)

            max_selection_classification = memory_bank_classification.sample_measurement_ordering_ascending.max().item()
            max_selection_asif = memory_bank_asif.sample_measurement_ordering_ascending.max().item()
            max_selection = min(max_selection_classification, max_selection_asif)
            selections = np.arange(max_selection)
            all_selections_ascending = np.zeros(max_selection)
            noisy_selections_ascending = np.zeros(max_selection)
            for selection in range(max_selection):
                is_selected_classification = \
                    memory_bank_classification.sample_measurement_ordering_ascending < (selection + 1)
                is_selected_asif = memory_bank_asif.sample_measurement_ordering_ascending < (selection + 1)
                is_selected = torch.logical_and(is_selected_classification, is_selected_asif)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                all_selections_ascending[selection] = is_selected.sum().item()
                noisy_selections_ascending[selection] = is_false_and_selected.sum().item()

            ax.plot(selections, noisy_selections_ascending,
                    label='Bad - Classification (A), ASIF (A)', linestyle='dashed')
            ax.plot(selections, all_selections_ascending, label='All - Classification (A), ASIF (A)')

            # Output ascending class vs descending ASIF combination...
            max_selection_classification = memory_bank_classification.sample_measurement_ordering_ascending.max().item()
            max_selection_asif = memory_bank_asif.sample_measurement_ordering_descending.max().item()
            max_selection = min(max_selection_classification, max_selection_asif)
            selections = np.arange(max_selection)
            all_selections_ascending = np.zeros(max_selection)
            noisy_selections_ascending = np.zeros(max_selection)
            for selection in range(max_selection):
                is_selected_classification = \
                    memory_bank_classification.sample_measurement_ordering_ascending < (selection + 1)
                is_selected_asif = memory_bank_asif.sample_measurement_ordering_descending < (selection + 1)
                is_selected = torch.logical_and(is_selected_classification, is_selected_asif)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                all_selections_ascending[selection] = is_selected.sum().item()
                noisy_selections_ascending[selection] = is_false_and_selected.sum().item()

            ax.plot(selections, noisy_selections_ascending,
                    label='Bad - Classification (A), ASIF (D)', linestyle='dashed')
            ax.plot(selections, all_selections_ascending, label='All - Classification (A), ASIF (D)')

            # Output ascending class vs ascending ASIF combination...
            max_selection_classification = \
                memory_bank_classification.sample_measurement_ordering_descending.max().item()
            max_selection_asif = memory_bank_asif.sample_measurement_ordering_ascending.max().item()
            max_selection = min(max_selection_classification, max_selection_asif)
            selections = np.arange(max_selection)
            all_selections_ascending = np.zeros(max_selection)
            noisy_selections_ascending = np.zeros(max_selection)
            for selection in range(max_selection):
                is_selected_classification = \
                    memory_bank_classification.sample_measurement_ordering_descending < (selection + 1)
                is_selected_asif = memory_bank_asif.sample_measurement_ordering_ascending < (selection + 1)
                is_selected = torch.logical_and(is_selected_classification, is_selected_asif)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                all_selections_ascending[selection] = is_selected.sum().item()
                noisy_selections_ascending[selection] = is_false_and_selected.sum().item()

            ax.plot(selections, noisy_selections_ascending,
                    label='Bad - Classification (D), ASIF (A)', linestyle='dashed')
            ax.plot(selections, all_selections_ascending, label='All - Classification (D), ASIF (A)')

            # Output ascending class vs descending ASIF combination...
            max_selection_classification = \
                memory_bank_classification.sample_measurement_ordering_descending.max().item()
            max_selection_asif = memory_bank_asif.sample_measurement_ordering_descending.max().item()
            max_selection = min(max_selection_classification, max_selection_asif)
            selections = np.arange(max_selection)
            all_selections_ascending = np.zeros(max_selection)
            noisy_selections_ascending = np.zeros(max_selection)
            for selection in range(max_selection):
                is_selected_classification = \
                    memory_bank_classification.sample_measurement_ordering_descending < (selection + 1)
                is_selected_asif = memory_bank_asif.sample_measurement_ordering_descending < (selection + 1)
                is_selected = torch.logical_and(is_selected_classification, is_selected_asif)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                all_selections_ascending[selection] = is_selected.sum().item()
                noisy_selections_ascending[selection] = is_false_and_selected.sum().item()

            ax.plot(selections, noisy_selections_ascending,
                    label='Bad - Classification (D), ASIF (D)', linestyle='dashed')
            ax.plot(selections, all_selections_ascending, label='All - Classification (D), ASIF (D)')
            ax.legend()

        fig.savefig(self.trial_data[config][trial_index].log_file_path.replace(
            ".txt",
            f"_loss_selection_epoch_{epoch}.png"))

    def _write_correct_selection_data_chart(self,
                                            penalty: float,
                                            epoch: int,
                                            trial_index: int,
                                            config: str):
        """
        Outputs a chart of the total true selections vs selection count. If false labels are selected, the total true
        count is penalised by the number of false selections times a penalty factor.
        :param penalty: The penalty factor to apply to false selections.
        :param epoch: The current epoch.
        :param trial_index: The current trial.
        :param config: The current configuration.
        """
        if epoch == 0:
            return

        fig = plt.figure(figsize=(24, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f'Total Penalised True Picks vs Selection Count (Penalty Factor {penalty})', fontsize=12)
        ax.set_ylim([0, 50000.])

        memory_bank_classification = self.trial_data[config][trial_index].loss_memory_bank_classification
        max_selection = memory_bank_classification.sample_measurement_ordering_ascending.max().item()
        true_labelled = (memory_bank_classification.sample_true_labels == memory_bank_classification.sample_labels)
        false_labelled = (memory_bank_classification.sample_true_labels != memory_bank_classification.sample_labels)

        selections = np.arange(max_selection)
        true_selections_ascending = np.zeros(max_selection)
        true_selections_descending = np.zeros(max_selection)
        for selection in range(max_selection):
            is_selected = memory_bank_classification.sample_measurement_ordering_ascending < (selection + 1)
            is_true_and_selected = torch.logical_and(is_selected, true_labelled)
            is_false_and_selected = torch.logical_and(is_selected, false_labelled)
            true_selections_ascending[selection] = \
                is_true_and_selected.sum().item() - is_false_and_selected.sum().item() * penalty

            is_selected = memory_bank_classification.sample_measurement_ordering_descending < (selection + 1)
            is_true_and_selected = torch.logical_and(is_selected, true_labelled)
            is_false_and_selected = torch.logical_and(is_selected, false_labelled)
            true_selections_descending[selection] = \
                is_true_and_selected.sum().item() - is_false_and_selected.sum().item() * penalty

        ax.plot(selections, true_selections_ascending, label="Ascending Classification Loss")
        ax.plot(selections, true_selections_descending, label="Descending Classification Loss")

        # Output the ASIF data...
        if self.configurations[config].use_asif:
            memory_bank_asif = self.trial_data[config][trial_index].loss_memory_bank_asif
            max_selection = memory_bank_asif.sample_measurement_ordering_ascending.max().item()
            true_labelled = (memory_bank_asif.sample_true_labels == memory_bank_asif.sample_labels)
            false_labelled = (memory_bank_asif.sample_true_labels != memory_bank_asif.sample_labels)

            selections = np.arange(max_selection)
            true_selections_ascending = np.zeros(max_selection)
            true_selections_descending = np.zeros(max_selection)
            for selection in range(max_selection):
                is_selected = memory_bank_asif.sample_measurement_ordering_ascending < (selection + 1)
                is_true_and_selected = torch.logical_and(is_selected, true_labelled)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                true_selections_ascending[selection] = \
                    is_true_and_selected.sum().item() - is_false_and_selected.sum().item() * penalty

                is_selected = memory_bank_asif.sample_measurement_ordering_descending < (selection + 1)
                is_true_and_selected = torch.logical_and(is_selected, true_labelled)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                true_selections_descending[selection] = \
                    is_true_and_selected.sum().item() - is_false_and_selected.sum().item() * penalty

            ax.plot(selections, true_selections_ascending, label="Ascending ASIF Loss")
            ax.plot(selections, true_selections_descending, label="Descending ASIF Loss")

            # Output ascending class vs ascending ASIF combination...
            max_selection_classification = memory_bank_classification.sample_measurement_ordering_ascending.max().item()
            max_selection_asif = memory_bank_asif.sample_measurement_ordering_ascending.max().item()
            max_selection = min(max_selection_classification, max_selection_asif)
            selections = np.arange(max_selection)
            class_and_asif_ascending = np.zeros(max_selection)
            not_asif_or_class_ascending = np.zeros(max_selection)
            for selection in range(max_selection):
                is_selected_classification = \
                    memory_bank_classification.sample_measurement_ordering_ascending < (selection + 1)
                is_selected_asif = memory_bank_asif.sample_measurement_ordering_ascending < (selection + 1)

                is_selected = torch.logical_and(is_selected_classification, is_selected_asif)
                is_true_and_selected = torch.logical_and(is_selected, true_labelled)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                class_and_asif_ascending[selection] = \
                    is_true_and_selected.sum().item() - is_false_and_selected.sum().item() * penalty

                is_selected = torch.logical_and(torch.logical_not(is_selected_asif),
                                                torch.logical_not(is_selected_classification))
                is_true_and_selected = torch.logical_and(is_selected, true_labelled)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                not_asif_or_class_ascending[selection] = \
                    is_true_and_selected.sum().item() - is_false_and_selected.sum().item() * penalty

            ax.plot(selections, class_and_asif_ascending,
                    label="A Classification and A ASIF",
                    linestyle='dashed')
            ax.plot(selections, not_asif_or_class_ascending,
                    label="not A ASIF and not A Classification",
                    linestyle='dashed')

            # Output ascending class vs descending ASIF combination...
            max_selection_classification = memory_bank_classification.sample_measurement_ordering_ascending.max().item()
            max_selection_asif = memory_bank_asif.sample_measurement_ordering_descending.max().item()
            max_selection = min(max_selection_classification, max_selection_asif)
            selections = np.arange(max_selection)
            class_and_asif_ascending = np.zeros(max_selection)
            not_asif_or_class_ascending = np.zeros(max_selection)
            for selection in range(max_selection):
                is_selected_classification = \
                    memory_bank_classification.sample_measurement_ordering_ascending < (selection + 1)
                is_selected_asif = memory_bank_asif.sample_measurement_ordering_descending < (selection + 1)

                is_selected = torch.logical_and(is_selected_classification, is_selected_asif)
                is_true_and_selected = torch.logical_and(is_selected, true_labelled)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                class_and_asif_ascending[selection] = \
                    is_true_and_selected.sum().item() - is_false_and_selected.sum().item() * penalty

                is_selected = torch.logical_and(torch.logical_not(is_selected_asif),
                                                torch.logical_not(is_selected_classification))
                is_true_and_selected = torch.logical_and(is_selected, true_labelled)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                not_asif_or_class_ascending[selection] = \
                    is_true_and_selected.sum().item() - is_false_and_selected.sum().item() * penalty

            ax.plot(selections, class_and_asif_ascending,
                    label="A Classification and D ASIF",
                    linestyle='dashed')
            ax.plot(selections, not_asif_or_class_ascending,
                    label="not D ASIF and not A Classification",
                    linestyle='dashed')

            # Output ascending class vs ascending ASIF combination...
            max_selection_classification = \
                memory_bank_classification.sample_measurement_ordering_descending.max().item()
            max_selection_asif = memory_bank_asif.sample_measurement_ordering_ascending.max().item()
            max_selection = min(max_selection_classification, max_selection_asif)
            selections = np.arange(max_selection)
            class_and_asif_ascending = np.zeros(max_selection)
            not_asif_or_class_ascending = np.zeros(max_selection)
            for selection in range(max_selection):
                is_selected_classification = \
                    memory_bank_classification.sample_measurement_ordering_descending < (selection + 1)
                is_selected_asif = memory_bank_asif.sample_measurement_ordering_ascending < (selection + 1)

                is_selected = torch.logical_and(is_selected_classification, is_selected_asif)
                is_true_and_selected = torch.logical_and(is_selected, true_labelled)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                class_and_asif_ascending[selection] = \
                    is_true_and_selected.sum().item() - is_false_and_selected.sum().item() * penalty

                is_selected = torch.logical_and(torch.logical_not(is_selected_asif),
                                                torch.logical_not(is_selected_classification))
                is_true_and_selected = torch.logical_and(is_selected, true_labelled)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                not_asif_or_class_ascending[selection] = \
                    is_true_and_selected.sum().item() - is_false_and_selected.sum().item() * penalty

            ax.plot(selections, class_and_asif_ascending,
                    label="D Classification and A ASIF",
                    linestyle='dashed')
            ax.plot(selections, not_asif_or_class_ascending,
                    label="not A ASIF and not D Classification",
                    linestyle='dashed')

            # Output ascending class vs descending ASIF combination...
            max_selection_classification = \
                memory_bank_classification.sample_measurement_ordering_descending.max().item()
            max_selection_asif = memory_bank_asif.sample_measurement_ordering_descending.max().item()
            max_selection = min(max_selection_classification, max_selection_asif)
            selections = np.arange(max_selection)
            class_and_asif_ascending = np.zeros(max_selection)
            not_asif_or_class_ascending = np.zeros(max_selection)
            for selection in range(max_selection):
                is_selected_classification = \
                    memory_bank_classification.sample_measurement_ordering_descending < (selection + 1)
                is_selected_asif = memory_bank_asif.sample_measurement_ordering_descending < (selection + 1)

                is_selected = torch.logical_and(is_selected_classification, is_selected_asif)
                is_true_and_selected = torch.logical_and(is_selected, true_labelled)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                class_and_asif_ascending[selection] = \
                    is_true_and_selected.sum().item() - is_false_and_selected.sum().item() * penalty

                is_selected = torch.logical_and(torch.logical_not(is_selected_asif),
                                                torch.logical_not(is_selected_classification))
                is_true_and_selected = torch.logical_and(is_selected, true_labelled)
                is_false_and_selected = torch.logical_and(is_selected, false_labelled)
                not_asif_or_class_ascending[selection] = \
                    is_true_and_selected.sum().item() - is_false_and_selected.sum().item() * penalty

            ax.plot(selections, class_and_asif_ascending,
                    label="D Classification and D ASIF",
                    linestyle='dashed')
            ax.plot(selections, not_asif_or_class_ascending,
                    label="not D ASIF and not D Classification",
                    linestyle='dashed')

        ax.legend()
        fig.savefig(self.trial_data[config][trial_index].log_file_path.replace(
            ".txt",
            f"_penalty_{penalty}_selection_epoch_{epoch}.png"))

    def _write_sample_data_histogram(self,
                                     epoch: int,
                                     data: Dict[int, Dict[str, float]],
                                     trial_index: int,
                                     config: str):
        """
        Takes a datum calculated for each individual sample and saves a histogram. The datum is specified by `datum`
        and can be found in the `data` structure.
        :param epoch: The current epoch.
        :param data: The data to plot.
        :param config: The experiment configuration to save.
        :param trial_index: The index of the trial to save.
        """
        # noinspection PyUnresolvedReferences
        bins = np.linspace(8., 9., 200)

        index = 0
        fig_bars = plt.figure(figsize=(8, 12))
        fig_pdf_bottom = plt.figure(figsize=(8, 12))
        fig_pdf_top = plt.figure(figsize=(8, 12))
        for label_id in range(10):
            data_to_plot = []
            for index_ in data:
                record_ = data[index_]
                if record_['label_id'] == label_id:
                    data_to_plot.append((record_['if_loss'], record_['label_wrong']))

            right_data = [data_to_plot[i][0] for i in range(len(data_to_plot)) if not data_to_plot[i][1]]
            wrong_data = [data_to_plot[i][0] for i in range(len(data_to_plot)) if data_to_plot[i][1]]

            index += 1
            ax = fig_bars.add_subplot(5, 2, index)
            ax.set_title(f'Class {index - 1} IF Losses', fontsize=20)

            wrong_bars, _, _ = ax.hist(wrong_data, bins, alpha=0.5, label='Noisy Labels')
            right_bars, _, _ = ax.hist(right_data, bins, alpha=0.5, label='Clean Labels')
            ax.legend(loc='upper right')

            # Calculate PDF of chances of having wrong data in the bottom 'x' samples...
            pdf = [0.] * len(right_bars)
            total_wrongs = [0] * len(right_bars)
            total = [0] * len(right_bars)
            for i in range(len(bins) - 1):
                if i == 0:
                    total_wrongs[i] = wrong_bars[i]
                    total[i] = right_bars[i] + wrong_bars[i]
                else:
                    total_wrongs[i] = total_wrongs[i - 1] + wrong_bars[i]
                    total[i] = total[i - 1] + right_bars[i] + wrong_bars[i]

                if total[i] != 0:
                    pdf[i] = total_wrongs[i] / total[i]

            ax = fig_pdf_bottom.add_subplot(5, 2, index)
            ax.set_title(f'Class {index - 1} Bottom Wrong Selection PDF', fontsize=20)
            ax.plot(bins[1:], pdf)

            # Calculate PDF of chances of having wrong data in the top 'x' samples...
            pdf = [0.] * len(right_bars)
            total_wrongs = [0] * len(right_bars)
            total = [0] * len(right_bars)
            for i in range(len(bins) - 2, -1, -1):
                if i == len(bins) - 2:
                    total_wrongs[i] = wrong_bars[i]
                    total[i] = right_bars[i] + wrong_bars[i]
                else:
                    total_wrongs[i] = total_wrongs[i + 1] + wrong_bars[i]
                    total[i] = total[i + 1] + right_bars[i] + wrong_bars[i]

                if total[i] != 0:
                    pdf[i] = total_wrongs[i] / total[i]

            ax = fig_pdf_top.add_subplot(5, 2, index)
            ax.set_title(f'Class {index - 1} Top Wrong Selection PDF', fontsize=20)
            ax.plot(bins[1:], pdf)

        fig_bars.savefig(self.trial_data[config][trial_index].log_file_path.replace(
            ".txt",
            f"_asif_sample_histogram_epoch_{epoch}.png"))
        fig_pdf_bottom.savefig(self.trial_data[config][trial_index].log_file_path.replace(
            ".txt",
            f"_asif_bottom_wrong_pdf_epoch_{epoch}.png"))
        fig_pdf_top.savefig(self.trial_data[config][trial_index].log_file_path.replace(
            ".txt",
            f"_asif_top_wrong_pdf_epoch_{epoch}.png"))

    def _write(self, text: str, trial_index: int = -1, config: str = None):
        """Write text to the log.

        :argument text The text to write.
        :argument trial_index Optional. If provided, write to a particular trial and config's log.
        :argument config Optional. If provided, write to a particular trial and config's log.
        """
        assert (trial_index == -1) or (trial_index > -1 and config is not None), \
            "If trial is specified, the config must also be."

        if trial_index > -1:
            # Write to a specific trial's log...
            print("{0} t{1}: {2}".format(config, self.trial_data[config][trial_index].trial_index, text))
            self.trial_data[config][trial_index].log.write("{0}\n".format(text))
        elif config is not None:
            # Write to all trials in a specific config...
            print("{0}: {1}".format(config, text))
            for trial_data in self.trial_data[config]:
                trial_data.log.write("{0}\n".format(text))
        else:
            # Write to all trials...
            print(text)
            for _config in self.trial_data:
                for trial_data in self.trial_data[_config]:
                    trial_data.log.write("{0}\n".format(text))

    def _flush_log(self):
        """Force the log to write to the file."""
        for _config in self.trial_data:
            for trial_data in self.trial_data[_config]:
                trial_data.log.flush()


class _TrialData:
    """Contains data for a specific trial run."""

    def __init__(self,
                 config_name: str,
                 label: str,
                 trial_index: int,
                 parent: TrainerASIFLabelNoise):
        """Create a new trial data.

        :argument config_name The unique name for the training run.
        :argument label A human-readable name for this configuration.
        :argument trial_index The zero-based index of the trial this data pertains to.
        :argument parent Reference to the trainer that owns this data.
        """
        self.parent = parent
        self.label = label
        self.trial_index = trial_index
        self.state_file_name = "{0}_i{1}.pth.tar".format(config_name, trial_index)
        self.log_file_path = "{0}_i{1}.txt".format(config_name, trial_index)
        self.log = open(self.log_file_path, 'a+')
        self.config_name = config_name

        self.best_accuracy = 0.

        self.model: nn.Module = None
        self.if_heads: List[nn.Module] = None
        self.optimizers = {}
        self.cc_manager: ConstrastiveClusterer = None
        self.loss_memory_bank_classification: LossMemoryBank = None
        self.loss_memory_bank_asif: LossMemoryBank = None
        self.asif_energy_memory_bank: LossMemoryBank = None

        self.cls_memory_bank: SortingMemoryBank = None
        self.cls_entropy_memory_bank: SortingMemoryBank = None
        self.asif_loss_memory_bank: SortingMemoryBank = None
        self.entropy_head_memory_bank: SortingMemoryBank = None
        self.entropy_head_samplewise_memory_bank: SortingMemoryBank = None
        self.energy_head_samplewise_memory_bank: SortingMemoryBank = None

        self.if_entropy_memory_banks: List[LossMemoryBank] = None
        self.if_loss_memory_banks: List[LossMemoryBank] = None

        self.classification_criteria: Union[nn.CrossEntropyLoss, nn.NLLLoss] = None
        self.classification_per_sample_criteria: Union[nn.CrossEntropyLoss, nn.NLLLoss] = None

        self.gce_criteria: Union[GCELoss] = None
        self.phuber_criteria: Union[PHuberCrossEntropy] = None

        self.reset_trainer()

    def reset_trainer(self):
        self.best_accuracy = 0.
        self.model = self.parent.create_model(self.config_name)
        if self.parent.configurations[self.config_name].use_asif:
            self.if_heads = self.parent.create_if_heads(self.model, config=self.config_name)
        else:
            self.if_heads = []
        self.optimizers = {}
        self.cc_manager = ConstrastiveClusterer(
            handle_positive_cases=self.parent.configurations[self.config_name].cc_handle_positive_cases,
            handle_negative_cases=self.parent.configurations[self.config_name].cc_handle_negative_cases)
        self.loss_memory_bank_classification = LossMemoryBank(dataset_size=len(self.parent.training_data))
        self.loss_memory_bank_asif = LossMemoryBank(dataset_size=len(self.parent.training_data))
        self.asif_energy_memory_bank = LossMemoryBank(dataset_size=len(self.parent.training_data))

        self.cls_memory_bank = SortingMemoryBank(dataset_size=len(self.parent.training_data),
                                                 history_length=3,
                                                 history_forward=False)
        self.cls_entropy_memory_bank = SortingMemoryBank(dataset_size=len(self.parent.training_data),
                                                         history_length=3,
                                                         history_forward=False)
        self.asif_loss_memory_bank = SortingMemoryBank(dataset_size=len(self.parent.training_data),
                                                       history_length=3,
                                                       history_forward=False)
        self.entropy_head_memory_bank = SortingMemoryBank(dataset_size=len(self.parent.training_data),
                                                          history_length=3,
                                                          history_forward=False)
        self.entropy_head_samplewise_memory_bank = SortingMemoryBank(dataset_size=len(self.parent.training_data),
                                                                     history_length=3,
                                                                     history_forward=False)
        self.energy_head_samplewise_memory_bank = SortingMemoryBank(dataset_size=len(self.parent.training_data),
                                                                    history_length=3,
                                                                    history_forward=False)

        self.if_entropy_memory_banks = []
        self.if_loss_memory_banks = []
        for if_head_ in self.if_heads:
            self.if_entropy_memory_banks.append(LossMemoryBank(dataset_size=if_head_.out_features))
            self.if_loss_memory_banks.append(LossMemoryBank(dataset_size=if_head_.out_features))

        self.classification_criteria = \
            self.parent.create_classification_criteria(reduction='mean', config_name=self.config_name)
        self.classification_per_sample_criteria = \
            self.parent.create_classification_criteria(reduction='none', config_name=self.config_name)

        if self.parent.configurations[self.config_name].use_gce:
            self.gce_criteria = GCELoss(q=self.parent.configurations[self.config_name].gce_q,
                                        dataset_size=len(self.parent.training_data))

        if self.parent.configurations[self.config_name].use_phuber:
            self.phuber_criteria = PHuberCrossEntropy(tau=self.parent.configurations[self.config_name].phuber_tau)

    def create_optimizer(self,
                         parameters: Iterator[torch.nn.Parameter],
                         optimizer_type: OptimizerOption,
                         epoch: int,
                         lr_coefficient: float = 1.0):
        # Calculate the Learning Rate using the schedule used by Chen et al in "Beyond Class-Conditional Assumption:
        # A Primary Attempt to Combat Instance-Dependent Label Noise"
        factor = 0
        if epoch > 120:
            factor = 2
        elif epoch > 60:
            factor = 1
        lr = self.parent.configurations[self.config_name].lr * lr_coefficient * math.pow(0.2, factor)

        if optimizer_type == OptimizerOption.OPTIMIZER_SGD:
            return torch.optim.SGD(parameters,
                                   lr=lr,
                                   momentum=self.parent.configurations[self.config_name].sgd_momentum,
                                   weight_decay=self.parent.configurations[self.config_name].sgd_decay)
        elif optimizer_type == OptimizerOption.OPTIMIZER_ADAM:
            return torch.optim.Adam(parameters, lr=0.0001)
        else:
            raise ValueError('{0} is not a valid optimizer option.'.format(optimizer_type))

    def create_optimizers(self, epoch: int, force_creation: bool = False):
        """
        Sets up the optimizers needed for training. For some training methods, this occurs once for the first epoch.
        For other training methods, models are recreated with each epoch.
        :argument epoch The training epoch the optimizers are to be used for.
        :argument force_creation If true, always recreate the optimisers.
        """
        use_idn = self.parent.configurations[self.config_name].scheduler == SchedulerOption.SCHEDULER_IDN
        use_sgd = \
            self.parent.configurations[self.config_name].optimizer_fe == OptimizerOption.OPTIMIZER_SGD
        recreate_each_epoch = force_creation or (use_sgd and use_idn)

        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        if 'model_optimizer' not in self.optimizers or recreate_each_epoch:
            self.optimizers['model_optimizer'] = self.create_optimizer(
                parameters=model.get_fc_parameters(),
                optimizer_type=self.parent.configurations[self.config_name].optimizer_fe,
                epoch=epoch)

        use_sgd = \
            self.parent.configurations[self.config_name].optimizer_if == OptimizerOption.OPTIMIZER_SGD
        recreate_each_epoch = use_sgd and use_idn
        for i in range(len(self.if_heads)):
            name = 'if_feature_optimizer{0}'.format(i)
            if name not in self.optimizers or recreate_each_epoch:
                params = chain(model.get_shared_if_parameters(), self.if_heads[i].parameters())
                self.optimizers[name] = self.create_optimizer(
                    lr_coefficient=0.0001,
                    parameters=params,
                    optimizer_type=self.parent.configurations[self.config_name].optimizer_fe,
                    epoch=epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ASIF Label Noise Experiment')

    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--label_dir', type=str, help='Path to the label noise files')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to the dataset')
    parser.add_argument('--remediation_file', type=str, default=None, help='Path to the label remediation file')
    parser.add_argument('--configs_path', type=str, help='Path to the configuration data file.')
    parser.add_argument('--specific_configs', type=str, default=None, help='Comma-separated list of config indices.')
    parser.add_argument('--config_index', type=int, default=0, help='One-based index into the configuration table.')
    parser.add_argument('--config_count', type=int, default=1, help='Number of configs to run at once.')
    parser.add_argument('--config_skip', type=int, default=0, help='Skip the first X configs in the selected range')
    parser.add_argument('--output_charts', type=int, default=0, help='If 1, output histograms during training.')
    parser.add_argument('--trial_count', type=int, default=1, help='The number of simultaneous trials to attempt.')
    parser.add_argument('--use_cuda', type=int, default=1, help='If 1, attempt to use CUDA')

    args = parser.parse_args()

    _config_details = Configurator.load(args.configs_path)

    if args.specific_configs is not None:
        _config_indices = [int(index) for index in args.specific_configs.split(",")]
        _subset = []
        for _index in _config_indices:
            _subset.append(_config_details[_index - 1])
    elif args.config_index > 0:
        _config_indices = []
        for _i in range(args.config_count):
            _config_indices.append((args.config_index - 1) * args.config_count + _i + 1)

        _subset = []
        for _index in _config_indices:
            _subset.append(_config_details[_index - 1])

        if args.config_skip > 0:
            _subset = _subset[args.config_skip-1:]
    else:
        _subset = _config_details

    _lr = None
    _dataset = None
    _noise_method = None
    _noise_level = None
    _noise_level_index = None
    _label_file_dir = args.label_dir
    _trial_index = 0
    _desired_samples_per_label = 0
    _use_augmentation = None
    _model = None
    _technique = None
    _optimizer_fe = None
    _optimizer_if = None
    _loss = None
    _scheduler = None
    _nishi_lr_switch_iteration = None
    _iteration_count = None
    _use_unsupervised_asif = None
    _asif_start_epoch = None
    _asif_loss_coefficient = 1.
    _shared_if_head_layer_count = 0
    _overridden_feature_vector_size = None
    _if_reverse_private = True
    _if_reverse_public = False
    _class_pick_clean_indices_start_epoch = 1
    _class_pick_clean_indices = CleanSamplePickingOption.DO_NOT_PICK
    _class_pick_clean_indices_start_percentile = 0.
    _class_pick_clean_indices_stop_percentile = 100.
    _use_dgr = True
    _gce_q = None
    _gce_start_prune_epoch = None
    _sgd_decay = None
    _sgd_momentum = None
    _phuber_tau = None

    for _configuration in _subset:
        _configurations = {}

        _lr = _configuration.lr
        _dataset = parse_noisy_dataset_type(_configuration.dataset)
        _noise_method = _configuration.noise_method
        _noise_level = _configuration.noise_level
        _noise_level_index = _configuration.noise_level_index
        _use_augmentation = _configuration.use_augmentation
        _model = _configuration.model
        _technique = _configuration.technique
        _optimizer_fe = _configuration.optimizer_fe
        _optimizer_if = _configuration.optimizer_if
        _loss = _configuration.loss
        _scheduler = _configuration.scheduler
        _nishi_lr_switch_iteration = _configuration.nishi_lr_switch_iteration
        _iteration_count = _configuration.iteration_count
        _trial_index = _configuration.trial_index
        _desired_samples_per_label = _configuration.desired_samples_per_label
        _use_unsupervised_asif = _configuration.use_unsupervised_asif
        _asif_start_epoch = _configuration.asif_start_epoch
        _asif_loss_coefficient = _configuration.asif_loss_coefficient
        _shared_if_head_layer_count = _configuration.shared_if_head_layer_count
        _overridden_feature_vector_size = _configuration.overridden_feature_vector_size
        _if_reverse_private = _configuration.if_reverse_private
        _if_reverse_public = _configuration.if_reverse_public
        _cc_pick_clean_indices = _configuration.cc_pick_clean_indices
        _class_pick_clean_indices = _configuration.class_pick_clean_indices
        _class_pick_clean_indices_start_epoch = _configuration.class_pick_clean_indices_start_epoch
        _class_pick_clean_indices_start_percentile = _configuration.class_pick_clean_indices_start_percentile
        _class_pick_clean_indices_stop_percentile = _configuration.class_pick_clean_indices_stop_percentile
        _cc_handle_positive_cases = _configuration.cc_handle_positive_cases
        _cc_handle_negative_cases = _configuration.cc_handle_negative_cases
        _bad_label_picking_frequency = _configuration.bad_label_picking_frequency
        _bad_label_picking_sample_count = _configuration.bad_label_picking_sample_count
        _bad_label_picking_cycle_count = _configuration.bad_label_picking_cycle_count
        _use_dgr = _configuration.use_dgr
        _gce_q = _configuration.gce_q
        _gce_start_prune_epoch = _configuration.gce_start_prune_epoch
        _sgd_decay = _configuration.sgd_decay
        _sgd_momentum = _configuration.sgd_momentum
        _phuber_tau = _configuration.phuber_tau

        if _iteration_count == 0:
            _iteration_count = 150 * 391

        _cc_loss_coefficient = _configuration.cc_loss_coefficient

        if _noise_method != LabelNoiseMethod.LABEL_NOISE_NONE:
            description_noise_level_sentence = 'The noise level is set to {0}%. '.format(_noise_level)
        else:
            description_noise_level_sentence = ''

        if _desired_samples_per_label > 0:
            subset_sentence = \
                'Training is on a subset of the dataset, only {0} samples per label. '.format(
                    _desired_samples_per_label)
        else:
            subset_sentence = ''

        if _use_unsupervised_asif:
            unsupervised_sentence = \
                'ASIF is performed with a single group and done regardless of whether a label is available. '
        else:
            unsupervised_sentence = ''

        asif_start_sentence = f'ASIF starts on epoch {_asif_start_epoch}. '

        if _cc_handle_negative_cases and _cc_handle_positive_cases:
            cc_mode_str = "CCPN"
        elif _cc_handle_negative_cases:
            cc_mode_str = "CCN"
        elif _cc_handle_positive_cases:
            cc_mode_str = "CCP"
        else:
            cc_mode_str = "CC_"

        _config_subname = '{0}.{1}.{2}.{3}.{4}.{5}.{6}.{7}.{8}.{9}.{10}.{11}.{12}.{13}.{14}.{15}.LR{16}'.format(
            _model,
            _configuration.noise_method.get_short_name(),
            _noise_level,
            _optimizer_if.get_short_name(),
            _optimizer_fe.get_short_name(),
            _scheduler.get_short_name(),
            _loss.get_short_name(),
            _class_pick_clean_indices.get_short_name(),
            _class_pick_clean_indices_start_epoch,
            _class_pick_clean_indices_start_percentile,
            _class_pick_clean_indices_stop_percentile,
            _bad_label_picking_frequency,
            _bad_label_picking_sample_count,
            _bad_label_picking_cycle_count,
            'all' if _desired_samples_per_label == 0 else _desired_samples_per_label,
            _iteration_count,
            _lr)

        _config_details = {
            '{0}.asifcc.{1}.{2}.{3}.{4}.{5}.{6}.{7}.{8}.{9}.{10}.{11}.{12}.{13}'.format(
                _dataset.get_short_name(),
                _cc_loss_coefficient,
                _cc_pick_clean_indices,
                _config_subname,
                _asif_start_epoch,
                _asif_loss_coefficient,
                _shared_if_head_layer_count,
                _configuration.small_loss_cutoff,
                _if_reverse_private,
                _if_reverse_public,
                _overridden_feature_vector_size if _overridden_feature_vector_size is not None else "No",
                cc_mode_str,
                _use_dgr,
                "U" if _use_unsupervised_asif else "S"): {
                'label': 'Trains on {0} with ASIF and CC with {1}.'.format(_dataset.get_short_name().upper(),
                                                                           _noise_method.get_friendly_name()),
                'description': 'Trains on {0} with ASIF with {1}. '.format(_dataset.get_short_name().upper(),
                                                                           _noise_method.get_friendly_name()) +
                               description_noise_level_sentence + subset_sentence + unsupervised_sentence +
                               asif_start_sentence +
                               'This is designed to keep the identity feature ' +
                               'classifier in a state of maximum uncertainty. Contrastive clustering is applied ' +
                               'after an initial burn-in period. It is only applied samples with a loss below ' +
                               '{0}. The base model is {1}. '.format(_configuration.small_loss_cutoff, _model) +
                               'The FE optimizer used is {0}'.format(_optimizer_fe.get_short_name()),
                'use_asif': True,
                'use_gce': False,
                'use_phuber': False,
                'use_cc': True,
                'cc_start_epoch': 5,
                'true_label_estimate_cutoff': _configuration.small_loss_cutoff,
                'base_model_type': _model,
                'technique': 'ASIFCC',
            },
            '{0}.cc.{1}.{2}.{3}.{4}.{5}'.format(_dataset.get_short_name(),
                                                _cc_loss_coefficient,
                                                _cc_pick_clean_indices,
                                                _configuration.small_loss_cutoff,
                                                cc_mode_str,
                                                _config_subname): {
                'label': 'Trains on {0} with ASIF and CC with {1}.'.format(_dataset.get_short_name().upper(),
                                                                           _noise_method.get_friendly_name()),
                'description': 'Trains on {0} with ASIF with {1}. '.format(_dataset.get_short_name().upper(),
                                                                           _noise_method.get_friendly_name()) +
                               description_noise_level_sentence + subset_sentence +
                               'This is designed to keep the identity feature ' +
                               'classifier in a state of maximum uncertainty. Contrastive clustering is applied ' +
                               'after an initial burn-in period. It is only applied samples with a loss below ' +
                               '{0}. The base model is {1}. '.format(_configuration.small_loss_cutoff, _model) +
                               'The fe optimizer used is {0}'.format(_optimizer_fe.get_short_name()),
                'use_asif': False,
                'use_gce': False,
                'use_phuber': False,
                'use_cc': True,
                'cc_start_epoch': 5,
                'true_label_estimate_cutoff': _configuration.small_loss_cutoff,
                'base_model_type': _model,
                'technique': 'CC',
            },
            '{0}.asif.{1}.{2}.{3}.{4}.{5}.{6}.{7}.{8}.{9}.{10}'.format(
                    _dataset.get_short_name(),
                    _config_subname,
                    _asif_start_epoch,
                    _asif_loss_coefficient,
                    _shared_if_head_layer_count,
                    _configuration.small_loss_cutoff,
                    _if_reverse_private,
                    _if_reverse_public,
                    _use_dgr,
                    _overridden_feature_vector_size if _overridden_feature_vector_size is not None else "No",
                    "U" if _use_unsupervised_asif else "S"): {
                'label': 'Trains on {0} with ASIF with {1}.'.format(_dataset.get_short_name().upper(),
                                                                    _noise_method.get_friendly_name()),
                'description': 'Trains on {0} with ASIF with {1}. '.format(_dataset.get_short_name().upper(),
                                                                           _noise_method.get_friendly_name()) +
                               description_noise_level_sentence + subset_sentence + unsupervised_sentence +
                               asif_start_sentence +
                               'This is designed to keep the identity feature ' +
                               'classifier in a state of maximum uncertainty. The base model is {0}. '.format(_model) +
                               'The FE optimizer used is {0}'.format(_optimizer_fe.get_short_name()),
                'use_asif': True,
                'use_gce': False,
                'use_phuber': False,
                'use_cc': False,
                'cc_start_epoch': 5,
                'true_label_estimate_cutoff': _configuration.small_loss_cutoff,
                'base_model_type': _model,
                'technique': 'ASIF',
            },
            '{0}.base.{1}'.format(
                    _dataset.get_short_name(),
                    _config_subname): {
                'label': 'Trains on {0} without ASIF with {1}.'.format(_dataset.get_short_name().upper(),
                                                                       _noise_method.get_friendly_name()),
                'description': 'Trains on {0} without ASIF with {1}. '.format(_dataset.get_short_name().upper(),
                                                                              _noise_method.get_friendly_name()) +
                               description_noise_level_sentence + subset_sentence +
                               'This will serve as a baseline against which to ' +
                               'compare other techniques. The base model is {0}. '.format(_model) +
                               'The FE optimizer used is {0}'.format(_optimizer_fe.get_short_name()),
                'use_asif': False,
                'use_gce': False,
                'use_phuber': False,
                'use_cc': False,
                'cc_start_epoch': 5,
                'true_label_estimate_cutoff': _configuration.small_loss_cutoff,
                'base_model_type': _model,
                'technique': 'Base',
            },
            '{0}.phuber.{1}.{2}'.format(
                    _dataset.get_short_name(),
                    _config_subname,
                    _phuber_tau): {
                'label': 'Trains on {0} using partially Huberised (PHuber) cross-entropy loss with {1}.'.format(
                    _dataset.get_short_name().upper(),
                    _noise_method.get_friendly_name()),
                'description': 'Trains on {0} using partially Huberised (PHuber) cross-entropy loss with {1}.'.format(
                    _dataset.get_short_name().upper(),
                    _noise_method.get_friendly_name()) +
                    description_noise_level_sentence + subset_sentence +
                    'This will serve as an alternative technique against which to ' +
                    'compare other ASIF. The base model is {0}. '.format(_model) +
                    'The FE optimizer used is {0}'.format(_optimizer_fe.get_short_name()),
                'use_asif': False,
                'use_gce': False,
                'use_phuber': True,
                'use_cc': False,
                'cc_start_epoch': 5,
                'true_label_estimate_cutoff': _configuration.small_loss_cutoff,
                'base_model_type': _model,
                'technique': 'phuber',
            },
            '{0}.gce.{1}.{2}.{3}'.format(
                    _dataset.get_short_name(),
                    _config_subname,
                    _gce_q,
                    _gce_start_prune_epoch): {
                'label': 'Trains on {0} using Generalized cross entropy loss with {1}.'.format(
                    _dataset.get_short_name().upper(),
                    _noise_method.get_friendly_name()),
                'description': 'Trains on {0} using Generalized cross entropy loss with {1}.'.format(
                    _dataset.get_short_name().upper(),
                    _noise_method.get_friendly_name()) +
                    description_noise_level_sentence + subset_sentence +
                    'This will serve as an alternative technique against which to ' +
                    'compare other ASIF. The base model is {0}. '.format(_model) +
                    'The FE optimizer used is {0}'.format(_optimizer_fe.get_short_name()),
                'use_asif': False,
                'use_gce': True,
                'use_phuber': False,
                'use_cc': False,
                'cc_start_epoch': 5,
                'true_label_estimate_cutoff': _configuration.small_loss_cutoff,
                'base_model_type': _model,
                'technique': 'gce',
            },
        }

        remediation_file_path_ = None
        for _config_name in _config_details:
            if _config_details[_config_name]['technique'].upper() == _technique.upper():
                remediation_file_path_ = "{0}_i{1}_wrong_labels.csv".format(_config_name, _trial_index)
                _configurations[_config_name] = \
                    TrainerConfig(
                        lr=_lr,
                        config_name=_config_name,
                        label=_config_details[_config_name]['label'],
                        description=_config_details[_config_name]['description'],
                        class_pick_clean_indices=_class_pick_clean_indices,
                        use_asif=_config_details[_config_name]['use_asif'],
                        use_gce=_config_details[_config_name]['use_gce'],
                        use_phuber=_config_details[_config_name]['use_phuber'],
                        use_cc=_config_details[_config_name]['use_cc'],
                        cc_handle_positive_cases=_cc_handle_positive_cases,
                        cc_handle_negative_cases=_cc_handle_negative_cases,
                        cc_pick_clean_indices=_cc_pick_clean_indices,
                        cc_start_epoch=_config_details[_config_name]['cc_start_epoch'],
                        optimizer_fe=_optimizer_fe,
                        optimizer_if=_optimizer_if,
                        loss=_loss,
                        scheduler=_scheduler,
                        true_label_estimate_loss_cutoff=_config_details[_config_name]['true_label_estimate_cutoff'],
                        cc_loss_coefficient=_cc_loss_coefficient,
                        base_model_type=_config_details[_config_name]['base_model_type'],
                        nishi_lr_switch_iteration=_nishi_lr_switch_iteration,
                        total_iterations=_iteration_count,
                        asif_start_epoch=_asif_start_epoch,
                        asif_loss_coefficient=_asif_loss_coefficient,
                        shared_if_head_layer_count=_shared_if_head_layer_count,
                        overridden_feature_vector_size=_overridden_feature_vector_size,
                        if_reverse_private=_if_reverse_private,
                        if_reverse_public=_if_reverse_public,
                        class_pick_clean_indices_start_epoch=_class_pick_clean_indices_start_epoch,
                        class_pick_clean_indices_start_percentile=_class_pick_clean_indices_start_percentile,
                        class_pick_clean_indices_stop_percentile=_class_pick_clean_indices_stop_percentile,
                        bad_label_picking_frequency=_bad_label_picking_frequency,
                        bad_label_picking_sample_count=_bad_label_picking_sample_count,
                        bad_label_picking_cycle_count=_bad_label_picking_cycle_count,
                        use_dgr=_use_dgr,
                        gce_q=_gce_q,
                        gce_start_prune_epoch=_gce_start_prune_epoch,
                        sgd_decay=_sgd_decay,
                        sgd_momentum=_sgd_momentum,
                        phuber_tau=_phuber_tau
                    )

        if not os.path.exists(remediation_file_path_):
            remediation_file_path_ = None
        trainer = TrainerASIFLabelNoise(data_dir=args.data_dir,
                                        dataset_type=_dataset,
                                        remediation_file=remediation_file_path_,
                                        configurations=_configurations,
                                        use_cuda=args.use_cuda == 1,
                                        num_trials=args.trial_count,
                                        noise_method=_noise_method,
                                        noise_level=_noise_level,
                                        noise_level_index=_noise_level_index,
                                        samples_per_label=_desired_samples_per_label,
                                        label_file_dir=_label_file_dir,
                                        apply_data_augmentation=_use_augmentation,
                                        first_trial_index=_trial_index,
                                        use_unsupervised_asif=_use_unsupervised_asif,
                                        output_charts=args.output_charts == 1)
        trainer.run_training()
