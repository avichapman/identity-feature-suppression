import math

import torch
import torch.nn as nn


class PHuberCrossEntropy(nn.Module):
    """
    PyTorch implementation of the loss described in the paper
    "Can gradient clipping mitigate label noise?" in ILCR 2020.

    This code is sourced from the implementation found at:
    https://github.com/dmizr/phuber/tree/3b70eadd9bd1420047ada743ff5604eda48d63ac
    """

    def __init__(self, tau: float = 10) -> None:
        """
        Creates the loss.
        :param tau: clipping threshold, must be > 1
        """
        super().__init__()
        self.tau = tau

        # Probability threshold for the clipping
        self.prob_thresh = 1 / self.tau
        # Negative of the Fenchel conjugate of base loss at tau
        self.boundary_term = math.log(self.tau) + 1

        self.softmax = nn.Softmax(dim=1)

    def forward(self, logits: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
        p = self.softmax(logits)
        p = p[torch.arange(p.shape[0]), label_ids]

        loss = torch.empty_like(p)
        clip = p <= self.prob_thresh
        loss[clip] = -self.tau * p[clip] + self.boundary_term
        loss[~clip] = -torch.log(p[~clip])

        return torch.mean(loss)
