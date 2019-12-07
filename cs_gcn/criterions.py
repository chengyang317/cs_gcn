import torch.nn as nn


class GqaCrossEntropy(nn.Module):

    def __init__(self,):
        super().__init__()
        self.loss_l = nn.CrossEntropyLoss()

    def forward(self, logits, a_labels):
        loss = self.loss_l(logits, a_labels)
        correct = logits.max(1)[1] == a_labels
        acc = correct.sum().item() / logits.size(0)
        return loss, acc

