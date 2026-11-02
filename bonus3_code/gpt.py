"""The prescribed GPT, implemented from basic PyTorch layers."""

from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn


@dataclass(frozen=True)
class GPTConfig:
    vocabulary_size: int = 65
    context_length: int = 256
    width: int = 384
    heads: int = 6
    layers: int = 6
    dropout: float = 0.2


class TokenPositionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_length = config.context_length
        self.token = nn.Embedding(config.vocabulary_size, config.width)
        self.position = nn.Embedding(config.context_length, config.width)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids, position_ids=None):
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("Expected (B,T) int64 token IDs.")
        length = token_ids.shape[1]
        if not 1 <= length <= self.context_length:
            raise ValueError("Sequence length must be between 1 and 256.")
        if position_ids is None:
            position_ids = torch.arange(length, device=token_ids.device)
        elif (position_ids.shape != (length,) or position_ids.dtype != torch.long
              or position_ids.device != token_ids.device):
            raise ValueError("Explicit position IDs must be length-T int64 IDs on the input device.")
        return self.dropout(self.token(token_ids) + self.position(position_ids).unsqueeze(0))


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.width % config.heads:
            raise ValueError("Embedding width must be divisible by the number of heads.")
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.projection = nn.Linear(config.width, config.width, bias=False)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(
            config.context_length, config.context_length, dtype=torch.bool)), persistent=False)

    def forward(self, hidden, return_attention=False):
        batch, length, width = hidden.shape
        query, key, value = self.qkv(hidden).split(width, dim=-1)
        query = query.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        key = key.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        value = value.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) * (self.head_width ** -0.5)
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = self.attention_dropout(weights) @ value
        combined = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        output = self.projection(combined)
        return (output, weights) if return_attention else output


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width, eps=1e-5, bias=False)
        self.attention = CausalMultiHeadAttention(config)
        self.attention_residual_dropout = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.width, eps=1e-5, bias=False)
        self.ffn = nn.Sequential(nn.Linear(config.width, 4 * config.width, bias=False),
                                 nn.GELU(), nn.Linear(4 * config.width, config.width, bias=False))
        self.ffn_residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, return_attention=False):
        attended = self.attention(self.norm1(hidden), return_attention=return_attention)
        if return_attention:
            attended, weights = attended
        residual = hidden + self.attention_residual_dropout(attended)
        output = residual + self.ffn_residual_dropout(self.ffn(self.norm2(residual)))
        return (output, weights) if return_attention else output


class GPT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or GPTConfig()
        self.embeddings = TokenPositionEmbedding(self.config)
        self.blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.layers)])
        self.final_norm = nn.LayerNorm(self.config.width, eps=1e-5, bias=False)
        self.output = nn.Linear(self.config.width, self.config.vocabulary_size, bias=False)
        # Tie before the subsequent prescribed initialization, not by copying.
        self.output.weight = self.embeddings.token.weight

    def forward(self, token_ids, position_ids=None, return_attention=False):
        hidden = self.embeddings(token_ids, position_ids)
        attentions = []
        for block in self.blocks:
            if return_attention:
                hidden, weights = block(hidden, return_attention=True)
                attentions.append(weights)
            else:
                hidden = block(hidden)
        logits = self.output(self.final_norm(hidden))
        return (logits, attentions) if return_attention else logits


def build_gpt(config=None):
    from paths import SEED
    torch.manual_seed(SEED)
    model = GPT(config)
    # parameters() deduplicates the tied tensor, so initialize it only once.
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        else:
            nn.init.ones_(parameter)
    for block in model.blocks:
        nn.init.normal_(block.attention.projection.weight, mean=0.0, std=0.02 / math.sqrt(12))
        nn.init.normal_(block.ffn[2].weight, mean=0.0, std=0.02 / math.sqrt(12))
    return model


def parameter_fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
