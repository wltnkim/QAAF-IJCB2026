"""Concordance Correlation Coefficient loss for valence-arousal regression.

CCC (Lin, 1989) measures agreement between a prediction and a target, combining
correlation with a penalty for differences in mean and in scale:

    CCC = 2 * rho * s_x * s_y / (s_x^2 + s_y^2 + (m_x - m_y)^2)

where rho is the Pearson correlation, s the standard deviation and m the mean.
It is 1 for perfect agreement and 0 for none, so the loss is 1 - CCC.

Reference:
    L. I-Kuei Lin. "A concordance correlation coefficient to evaluate
    reproducibility." Biometrics 45(1), 1989.
"""
import torch
import torch.nn as nn


class CCCLoss(nn.Module):
    """1 - CCC, computed over the flattened prediction and target.

    Args:
        digitize_num: kept for call-site compatibility. Only the plain
            regression case (0 or 1) is implemented; this code never used the
            binned-classification variant.
        range: retained for signature compatibility, unused in the regression
            case.
        eps: numerical floor. It is added inside each square root and to every
            denominator, so a constant prediction gives a finite loss and a
            finite gradient rather than a NaN.

    Both inputs are cast to float32 before the statistics are computed, so the
    loss is stable under mixed-precision training.
    """

    def __init__(self, digitize_num=1, range=(-1.0, 1.0), eps=1e-8):
        super().__init__()
        if digitize_num not in (0, 1):
            raise NotImplementedError(
                "CCCLoss here implements the regression case only "
                f"(digitize_num 0 or 1), got {digitize_num}.")
        self.digitize_num = digitize_num
        self.range = range
        self.eps = eps

    def forward(self, x, y):
        x = x.to(torch.float32).reshape(-1)
        y = y.to(torch.float32).reshape(-1)

        x_m, y_m = torch.mean(x), torch.mean(y)
        vx, vy = x - x_m, y - y_m

        rho = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx * vx) + self.eps)
            * torch.sqrt(torch.sum(vy * vy) + self.eps)
            + self.eps)

        x_s = torch.sqrt(torch.var(x, unbiased=False) + self.eps)
        y_s = torch.sqrt(torch.var(y, unbiased=False) + self.eps)

        ccc = 2 * rho * x_s * y_s / (
            x_s * x_s + y_s * y_s + (x_m - y_m) * (x_m - y_m) + self.eps)
        return 1 - ccc
