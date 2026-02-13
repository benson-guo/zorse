import torch.nn as nn

class LMLoss(nn.Module):
    """
    Deepspeed allows for creating PipelineModule with
        loss_fn (callable, optional): Loss is computed ``loss = loss_fn(outputs, label)``
    Used as language modelling loss

    NOTE: This loss does not overflow and skip iterations in deepspeed
        ( Recommended over sum loss )
    """
    def __init__(self):
        super(LMLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        outputs = outputs.view(-1, outputs.size(-1))
        labels = labels.view(-1)
        loss = self.criterion(outputs, labels)
        return loss


class RegressionLoss(nn.Module):
    """
    Used for ViT regression
    """
    def __init__(self):
        super(RegressionLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, outputs, labels):
        outputs = outputs.squeeze(-1).mean(dim=1)
        loss = self.criterion(outputs, labels)
        return loss