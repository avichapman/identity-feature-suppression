import statistics


class AverageMeter:
    """Computes and stores an average and current value"""
    def __init__(self, output_sum: bool = False):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.output_sum = output_sum
        self.values = []
        self.reset()

    def get_final_value(self) -> float:
        """Gets the final result. This can be the average or the sum, depending on `output_sum`."""
        if self.output_sum:
            return self.sum
        else:
            return self.avg

    def get_spread(self) -> str:
        """Gets the final result. This can be the average or the sum, depending on `output_sum`."""
        if len(self.values) >= 1:
            mean = statistics.mean(self.values)
        else:
            mean = 0.
        if len(self.values) >= 2:
            stdev = statistics.stdev(self.values)
        else:
            stdev = 0.

        return "{mean:.4f} +/- {stdev:.4f}".format(mean=mean, stdev=stdev)

    def reset(self):
        r"""Resets the count and sum back to zero.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.values = []

    def update(self, val: float, n=1):
        r"""Sets the current value and updates the count and sum.
        Args:
            val: The value at the current moment.
            n: A multiplying factor.
        """
        for i in range(n):
            self.values.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
