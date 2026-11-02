"""The prescribed character bigram table, without hidden or position state."""

import torch
from torch import nn

from paths import SEED


class BigramModel(nn.Module):
    def __init__(self, vocabulary_size=65):
        super().__init__()
        self.table = nn.Embedding(vocabulary_size, vocabulary_size)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        return self.table(token_ids)


def build_bigram(vocabulary_size=65):
    torch.manual_seed(SEED)
    return BigramModel(vocabulary_size)
