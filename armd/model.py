"""The shared-representation network and its joint objective."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class SharedNet(nn.Module):
    """One encoder over the base block, two families of heads reading it asymmetrically.

    The susceptibility head sees ``[z, exposure]``; the causal heads see
    ``[z, severity]`` and never the exposure block. That asymmetry is the point of the
    architecture, so it is enforced here by construction rather than by convention: the
    forward signature makes it impossible to route the exposure block into the
    propensity model without editing this class.
    """

    def __init__(
        self,
        base_dim: int,
        expo_dim: int,
        sev_dim: int,
        n_agents: int,
        dims: Sequence[int] = (256, 128),
        rep_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = base_dim
        for h in dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, rep_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

        self.succ_head = nn.Linear(rep_dim + expo_dim, n_agents)
        cdim = rep_dim + sev_dim
        self.prop_head = nn.Linear(cdim, 1)
        self.y0_head = nn.Sequential(nn.Linear(cdim, rep_dim), nn.ReLU(), nn.Linear(rep_dim, 1))
        self.y1_head = nn.Sequential(nn.Linear(cdim, rep_dim), nn.ReLU(), nn.Linear(rep_dim, 1))

        self.base_dim, self.expo_dim, self.sev_dim = base_dim, expo_dim, sev_dim

    def represent(self, xb: torch.Tensor) -> torch.Tensor:
        return self.encoder(xb)

    def forward(self, xb: torch.Tensor, xe: torch.Tensor, xs: torch.Tensor):
        z = self.encoder(xb)
        succ_in = torch.cat([z, xe], dim=1) if xe.shape[1] else z
        caus_in = torch.cat([z, xs], dim=1) if xs.shape[1] else z
        return (
            self.succ_head(succ_in),
            self.prop_head(caus_in).squeeze(-1),
            self.y0_head(caus_in).squeeze(-1),
            self.y1_head(caus_in).squeeze(-1),
        )


_bce = nn.BCEWithLogitsLoss(reduction="none")


def joint_loss(outputs, ys, ms, a, yc, mc, causal_weight: float):
    """Masked susceptibility loss plus the weighted propensity and outcome losses.

    ``ms`` zeroes agents that were not tested for a culture, so an untested panel entry
    contributes nothing rather than being read as susceptible. ``mc`` does the same for
    cultures that are not valid causal index events.

    ``.clamp(min=1.0)`` on the denominators keeps an all-masked mini-batch from dividing
    by zero; such a batch contributes a zero loss, which is the intended no-op.
    """
    succ_logit, prop_logit, y0_logit, y1_logit = outputs
    succ = (_bce(succ_logit, ys) * ms).sum() / ms.sum().clamp(min=1.0)
    prop = (_bce(prop_logit, a) * mc).sum() / mc.sum().clamp(min=1.0)
    yhat = torch.where(a > 0.5, y1_logit, y0_logit)
    outcome = (_bce(yhat, yc) * mc).sum() / mc.sum().clamp(min=1.0)
    total = succ + causal_weight * (prop + outcome)
    return total, succ, prop, outcome
