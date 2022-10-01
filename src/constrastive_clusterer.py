import torch
import torch.nn as nn
from typing import Dict, List, Optional

from class_prototyper import ClassPrototyper


class ConstrastiveClusterer:
    """
    Manages constrastive clustering, including keeping track of class prototypes and calculating losses.

    With samples not suspected of having incorrect labels, the samples are encouraged to cluster around the centroid
    of all samples with that label.

    With samples suspected of having incorrect labels, the samples are repulsed from the centroid of the associated
    incorrect label.
    """

    UNKNOWN_IGNORE = 0
    """
    Specifies that unknown samples should not have any clustering behaviour.
    """

    UNKNOWN_CLUSTER = 1
    """
    Specifies that unknown samples should cluster into their own extra cluster.
    """

    UNKNOWN_REPEL = 2
    """
    Specifies that unknown samples should be repelled from all known clusters.
    """

    UNKNOWN_GRAVITATIONAL = 3
    """
    Attract unknown toward their closest cluster.
    """

    def __init__(self,
                 unknown_behaviour: int = UNKNOWN_IGNORE,
                 desired_interclass_distance: float = 2.0,
                 handle_positive_cases: bool = True,
                 handle_negative_cases: bool = True):
        """
        Create a new clusterer.

        :param unknown_behaviour: The way to treat unknown samples (UNKNOWN_IGNORE, UNKNOWN_REPEL, UNKNOWN_GRAVITATIONAL
         or UNKNOWN_CLUSTER)
        :param desired_interclass_distance The desired min distance between clusters.
        :param handle_positive_cases If true, apply clustering loss to samples not suspected to be incorrectly labelled.
        :param handle_negative_cases If true, apply anti-clustering loss to samples suspected of being incorrectly
        labelled.
        """
        assert unknown_behaviour in [self.UNKNOWN_IGNORE,
                                     self.UNKNOWN_REPEL,
                                     self.UNKNOWN_CLUSTER,
                                     self.UNKNOWN_GRAVITATIONAL], "Unknown value for 'unknown_behaviour'"

        self.unknown_behaviour = unknown_behaviour
        self.desired_interclass_distance = desired_interclass_distance
        self.handle_positive_cases = handle_positive_cases
        self.handle_negative_cases = handle_negative_cases

        # Prototypes of each class to encourage clustering around
        self.class_prototypes = {}  # type: Dict[int, torch.Tensor]

        # List of recently used feature vectors for each class
        self.recent_class_feature_vectors = {}  # type: Dict[int, List[torch.Tensor]]

        self.prototyper = ClassPrototyper(max_recent_feature_vector_count=50)

        # Before this epoch, do not perform constrastive clustering. This is to allow the feature vectors time to
        # mature...
        self.cc_start_epoch = 2

        # After this number of epochs, update the prototypes from the recent vector history...
        self.cc_update_interval = 1

        # Smoothing factor when updating the prototypes...
        self.clustering_momentum = 0.99

        # Used for calculating distance loss to increase or decrease distance. If the hinge labels passed in are '1',
        # The loss will encourage clustering. If they are -1, the loss will discourage it...
        self.hinge_loss_known = nn.HingeEmbeddingLoss(self.desired_interclass_distance, reduction='none')

        # Special case for unlabelled stuff. Make the distance greater...
        self.hinge_loss_unknown = nn.HingeEmbeddingLoss(self.desired_interclass_distance * 2, reduction='none')

    def calc_loss(self,
                  feature_vectors: torch.Tensor,
                  labels: torch.Tensor,
                  probably_right: torch.Tensor,
                  current_epoch: int) -> Optional[torch.Tensor]:
        """
        Calculates the constrastive clustering loss for the current training iteration and the mini batch provided.
        :param feature_vectors: A batch of feature vectors to add to the history.
        :param labels: The labels that correspond to the elements in the batch.
        :param probably_right: Selection mask. If 1, the sample's label can be trusted. If zero, it can't.
        :param current_epoch: The current training epoch.
        :return: A loss value, or None if it is too early to use contrastive clustering.
        """
        if current_epoch < self.cc_start_epoch:
            # Too early...
            return None

        if (current_epoch - self.cc_start_epoch) % self.cc_update_interval == 0:
            # It's time to update the prototypes...
            self._update_prototypes()

        known_loss = self._get_loss(feature_vectors[labels != -1],
                                    labels[labels != -1],
                                    probably_right[labels != -1],
                                    self.hinge_loss_known)
        if self.unknown_behaviour == self.UNKNOWN_REPEL:
            unknown_loss = self._get_loss(feature_vectors[labels == -1],
                                          labels[labels == -1],
                                          probably_right[labels != -1],
                                          self.hinge_loss_unknown)
        elif self.unknown_behaviour == self.UNKNOWN_GRAVITATIONAL:
            unknown_loss = self._get_gravitational_loss(feature_vectors[labels == -1])
        else:
            unknown_loss = None

        if known_loss is not None and unknown_loss is not None:
            return torch.cat((known_loss, unknown_loss), dim=0).mean()
        elif known_loss is not None:
            return known_loss.mean()
        elif unknown_loss is not None:
            return unknown_loss.mean()
        else:
            return None

    def _get_loss(self,
                  feature_vectors: torch.Tensor,
                  labels: torch.Tensor,
                  probably_right: torch.Tensor,
                  hinge_loss: nn.HingeEmbeddingLoss):
        batch_size = feature_vectors.size()[0]
        ones = torch.ones_like(labels)
        negatives = -torch.ones_like(labels)

        if batch_size == 0:
            return None

        # Now compare the vectors to their prototypes for known types and encourage diversity for unknown types...
        distances = []
        hinge_labels = []
        for _current_label in self.class_prototypes:
            prototype = self._stack_data_layers(self.class_prototypes[_current_label], batch_size)
            distances.append(torch.cdist(feature_vectors, prototype, p=2))

            hinge_labels_for_class = torch.zeros_like(labels)

            if self.handle_positive_cases:
                # If the labels are correct, but not for this cluster, repel from the centroid...
                correct_label_other_label = torch.logical_and(probably_right, labels != _current_label)
                hinge_labels_for_class[correct_label_other_label] = negatives[correct_label_other_label]

                # If the labels are correct and for this cluster, attract to the centroid...
                correct_label_same_label = torch.logical_and(probably_right, labels == _current_label)
                hinge_labels_for_class[correct_label_same_label] = ones[correct_label_same_label]

            if self.handle_negative_cases:
                # If the labels are incorrect and for this cluster, we know it can't be this cluster, so repel it...
                incorrect_label_same_label = torch.logical_and(probably_right == 0, labels == _current_label)
                hinge_labels_for_class[incorrect_label_same_label] = negatives[incorrect_label_same_label]

            hinge_labels.append(hinge_labels_for_class)

        distances = torch.cat(distances, dim=0).mean(dim=1)
        hinge_labels = torch.cat(hinge_labels, dim=0)
        return hinge_loss(distances, hinge_labels)

    def _get_gravitational_loss(self, feature_vectors: torch.Tensor):
        """Calculates the loss based on the idea that each vector should want to move towards its closest cluster
        centre."""

        batch_size = feature_vectors.size()[0]

        if batch_size == 0:
            return None

        # Calculate distances to each cluster centre. This will produce an NxK matrix with columns being the distance to
        # each of the K cluster centres...
        distances = []
        for _current_label in self.class_prototypes:
            prototype = self._stack_data_layers(self.class_prototypes[_current_label], batch_size)
            distances.append(torch.cdist(feature_vectors, prototype, p=2).mean(dim=1))
        distances = torch.stack(distances).transpose(1, 0)

        # Take the smallest distance as the distance to the nearest cluster...
        distances = distances.min(dim=1)[0]

        # Add sqrt to ensure loss fall-off with distance...
        distances = torch.pow(distances, 0.5)

        # Return the mean for the entire batch...
        return distances

    def add_feature_vectors_to_history(self,
                                       feature_vectors: torch.Tensor,
                                       labels: torch.Tensor,
                                       probably_right: torch.Tensor):
        """
        Adds the feature vectors in a given batch to the history for each class, based on the values in labels.
        :param feature_vectors: A batch of feature vectors to add to the history.
        :param labels: The labels that correspond to the elements in the batch.
        :param probably_right: Selection mask. If 1, the sample's label can be trusted. If zero, it can't.
        """
        assert feature_vectors.size()[0] == labels.size()[0], "Feature vectors and labels must have same batch size."

        if self.unknown_behaviour == self.UNKNOWN_CLUSTER:
            selection_mask = probably_right
        else:
            selection_mask = torch.logical_and(probably_right, labels != -1)

        self.prototyper.add_feature_vectors_to_history(feature_vectors[selection_mask], labels[selection_mask])

    @staticmethod
    def _stack_data_layers(layer: torch.Tensor, layer_count: int) -> torch.Tensor:
        """
        Takes a tensor and stacks copies of it with 'layer_count' layers.
        Arguments:
            layer: A 2d tensor to be stacked.
        """
        assert len(layer.size()) == 1, "Layer is assumed to be 1d"
        layer = layer.unsqueeze(0)

        data_cube = []
        for _ in range(layer_count):
            data_cube.append(layer)
        data_cube = torch.cat(data_cube, dim=0)
        return data_cube

    def _update_prototypes(self):
        """
        Updates the prototypes for each class, based on the stored history of recent feature vectors.

        If this is the first time, we simply take the mean of the recent vectors for each feature.
        If this is an update, we do the same, but then apply a smoothing value so that the prototype doesn't suddenly
        jump.
        """
        for _class in self.prototyper.recent_class_feature_vectors:
            new_prototype = self.prototyper.calc_prototype_for_class(_class)

            if _class not in self.class_prototypes:
                self.class_prototypes[_class] = new_prototype
            else:
                previous_prototype = self.class_prototypes[_class]
                self.class_prototypes[_class] = self.clustering_momentum * previous_prototype + \
                    (1 - self.clustering_momentum) * new_prototype
