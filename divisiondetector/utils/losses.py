import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        """
        Combines BCEWithLogitsLoss and Dice Loss.

        Args:
            bce_weight (float): Weight for the BCE loss component.
            dice_weight (float): Weight for the Dice loss component.
            smooth (float): Smoothing term to avoid division by zero in Dice loss.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): The raw output from the model (before sigmoid). Shape: (N, *).
            targets (Tensor): Ground truth binary labels. Shape: (N, *).
        """
        # Compute BCE with Logits Loss
        bce = self.bce_loss(logits, targets)

        # Apply sigmoid to logits for dice computation
        # probs = torch.sigmoid(logits)
        probs = logits
        
        # Flatten tensors for Dice calculation
        probs_flat = probs.view(-1)
        targets_flat = targets.contiguous().view(-1)
        
        # Compute Dice Loss
        intersection = (probs_flat * targets_flat).sum()
        dice = 1 - (2. * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        # Combine losses
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        return total_loss