import torch
import torch.nn as nn


class JudoLoss(nn.Module):
    """
    Binary loss for one throw-attempt prediction per clip.
    
    Positve class refers to the clips with throw attempts.
    Need to weight loss since more examplse of throw attempt
    clips than no-throw attempt clips
    """

    def __init__(
        self,
        positive_class_weight: float,
    ) -> None:
        
        super().__init__()

        if positive_class_weight <= 0.0:
            raise ValueError(
                "positive_class_weight must be greater than zero"
            )

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [positive_class_weight],
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                "Logits and targets must have matching shapes, "
                f"got {tuple(logits.shape)} and "
                f"{tuple(targets.shape)}"
            )

        return self.bce(
            logits,
            targets.to(dtype=torch.float32),
        )
