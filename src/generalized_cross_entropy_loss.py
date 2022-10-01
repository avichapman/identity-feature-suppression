import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np


class GCELoss(nn.Module):
    """
    PyTorch implementation of the loss described in the paper
    "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels" in NIPS 2018.

    This code is sourced from Alan Chou's implementation found at https://github.com/AlanChou/Truncated-Loss

    To use this loss, apply like any other until you are ready to begin pruning. The paper describes this point as after
    40 epochs for CIFAR10 using SGD with 0.9 momentum, a weight decay of 10−4 and learning rate of 0.01. When that point
    arrives, call `update_weight` with the training set logits and class labels. The pruning factors for each sample
    will be updated. Subsequent use of the loss will result in some samples being discounted or eliminated from the
    training process.
    """

    def __init__(self, q: float = 0.7, k: float = 0.5, dataset_size: int = 50000):
        """
        Create a new loss.
        :param q: The `q` coefficient as described in Equation 6 in the paper. (0 < q <= 1)
        :param k: Truncation factor as described in Equation 9 in the paper. (0 < k < 1)
        :param dataset_size: The size of the training dataset. Used to store pruning factors for each sample.
        """
        super(GCELoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(dataset_size, 1), requires_grad=False)

    def forward(self, logits: torch.Tensor, label_ids: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns a scalar loss.
        :param logits: A mini batch of outputs from the classifier.
        :param label_ids: The class labels for each logit in the mini batch.
        :param sample_ids: The unique index of each sample that produced the logits in the mini batch.
        :return: A scalar loss.
        """
        p = functional.softmax(logits, dim=1)
        y_g = torch.gather(p, 1, torch.unsqueeze(label_ids, 1))

        loss = ((1 - (y_g ** self.q)) / self.q) * self.weight[sample_ids] - ((1 - (self.k ** self.q)) / self.q) * \
            self.weight[sample_ids]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits: torch.Tensor, label_ids: torch.Tensor, sample_ids: torch.Tensor):
        """
        Updates the weighting applied to each sample in the training set.

        This will result in certain samples having their contributions to the training process reduced or eliminated.
        :param logits: A mini batch of outputs from the classifier.
        :param label_ids: The class labels for each logit in the mini batch.
        :param sample_ids: The unique index of each sample that produced the logits in the mini batch.
        """
        p = functional.softmax(logits, dim=1)
        y_g = torch.gather(p, 1, torch.unsqueeze(label_ids, 1))
        l_q = ((1 - (y_g ** self.q)) / self.q)
        l_q_k = np.repeat(((1 - (self.k ** self.q)) / self.q), label_ids.size(0))
        l_q_k = torch.from_numpy(l_q_k).type(torch.float).cuda()
        l_q_k = torch.unsqueeze(l_q_k, 1)

        condition = torch.gt(l_q_k, l_q)
        self.weight[sample_ids] = condition.type(torch.float)
