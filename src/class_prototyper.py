import torch
import numpy as np
from typing import Dict, List


class ClassPrototyper:
    """
    Collects feature vectors from different classes and calculates a centroid.
    """

    def __init__(self, max_recent_feature_vector_count: int):
        """
        Create a new prototyper.

        :param max_recent_feature_vector_count: The maximum number of feature vectors to keep in the history.
        """

        # Keep up to this many feature vectors per class
        self.max_recent_feature_vector_count = max_recent_feature_vector_count

        # List of recently used feature vectors for each class
        self.recent_class_feature_vectors = {}  # type: Dict[int, List[torch.Tensor]]

    def clear_history(self):
        """Clears any previous feature vector history."""
        self.recent_class_feature_vectors.clear()

    def to_numpy(self) -> np.ndarray:
        output = None
        for class_id in self.recent_class_feature_vectors:
            class_ids = torch.ones((len(self.recent_class_feature_vectors[class_id]), 1)) * class_id
            vectors = torch.stack(self.recent_class_feature_vectors[class_id])
            stack = torch.cat((class_ids.cuda(), vectors), dim=1)
            if output is None:
                output = stack
            else:
                output = torch.cat((output, stack), dim=0)
        return output.cpu().numpy()

    def add_feature_vectors_to_history(self, feature_vectors: torch.Tensor, labels: torch.Tensor):
        """
        Adds the feature vectors in a given batch to the history for each class, based on the values in labels.
        :param feature_vectors: A batch of feature vectors to add to the history.
        :param labels: The labels that correspond to the elements in the batch.
        """
        assert feature_vectors.size()[0] == labels.size()[0], "Feature vectors and labels must have same batch size."

        batch_size = feature_vectors.size()[0]

        for i in range(batch_size):
            _label = labels[i].item()

            # If this is the first time encountering this label...
            if _label not in self.recent_class_feature_vectors:
                self.recent_class_feature_vectors[_label] = []

            # Push the vector onto the queue...
            if len(self.recent_class_feature_vectors[_label]) >= self.max_recent_feature_vector_count:
                self.recent_class_feature_vectors[_label].pop(0)
            self.recent_class_feature_vectors[_label].append(feature_vectors[i].detach())

    def calc_prototype_for_class(self, class_index: int) -> torch.Tensor:
        """
        Calculates the prototype for a class, based on the stored history of recent feature vectors.

        :argument class_index The class index to return a prototype for.
        """
        return torch.mean(torch.stack(self.recent_class_feature_vectors[class_index]), dim=0)

    def calc_stdev_for_class(self, class_index: int) -> torch.Tensor:
        """
        Calculates the standard deviation for all feature vectors in class, based on the stored history of recent
        feature vectors.

        :argument class_index The class index to return a prototype for.
        """
        return torch.std(torch.stack(self.recent_class_feature_vectors[class_index]), dim=0)
