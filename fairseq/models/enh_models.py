# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import logging
import math
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II, MISSING, open_dict

from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask
from transformers import WavLMModel, WavLMForCTC
from typing import Dict, List, Optional, Tuple
from fairseq.data.dictionary import Dictionary
from fairseq.tasks.enhancement import (
    EnhancementConfig,
    EnhancementTask,
)
from fairseq import utils
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from torch.nn import TransformerEncoderLayer, TransformerEncoder, Linear
from fairseq.models.wav2vec.wav2vec2 import make_conv_pos 

@dataclass
class EnhTransformerConfig(FairseqDataclass):
    setup: str = field(
        default='tiny',
        metadata={"help": "Transformer size"},
    )
    n_fft: int = field(
        default=512,
        metadata={"help": "NFFT"},
    )
    activation: str = field(
        default="none",
        metadata={"help": "Activation function"},
    )
    log1p: bool = field(
        default=True,
        metadata={"help": "log1p"},
    )
    pos_enc: str = field(
        default='sinusoid',
        metadata={"help": "Positional encoding type"},
    )

@dataclass
class EnhRNNConfig(FairseqDataclass):
    rnn: str = field(
        default='LSTM',
        metadata={"help": "RNN"},
    )
    d_model: int = field(
        default=256,
        metadata={"help": "Model dimension"},
    )
    rnn_layers: int = field(
        default=3,
        metadata={"help": "Number of RNN layers"},
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate"},
    )
    bidirectional: bool = field(
        default=True,
        metadata={"help": "Bidirectional RNN"},
    )
    n_fft: int = field(
        default=512,
        metadata={"help": "NFFT"},
    )
    activation: str = field(
        default="none",
        metadata={"help": "Activation function"},
    )
    log1p: bool = field(
        default=True,
        metadata={"help": "log1p"},
    )

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

@register_model("enh_rnn", dataclass=EnhRNNConfig)
class EnhRNN(BaseFairseqModel):
    def __init__(
        self, 
        cfg: EnhRNNConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.rnn = getattr(torch.nn, cfg.rnn)(
            int(cfg.n_fft / 2 + 1),
            cfg.d_model,
            cfg.rnn_layers,
            batch_first=False,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional
        )
        if cfg.bidirectional:
            self.output_proj = Linear(cfg.d_model * 2, int(cfg.n_fft / 2 + 1))
        else:
            self.output_proj = Linear(cfg.d_model, int(cfg.n_fft / 2 + 1))
        self.activation = {
            "relu": torch.nn.functional.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "none": torch.nn.Identity()
        }[cfg.activation]

    @classmethod
    def build_model(cls, cfg: EnhRNNConfig, task: EnhancementTask):
        return EnhRNN(cfg)

    def forward(self, src_stft_mag):
        assert len(src_stft_mag.size()) == 4
        src_stft_mag = (src_stft_mag[:, 0, :, :]).transpose(0, 1)
        if self.cfg.log1p:
            input_stft = torch.log1p(src_stft_mag)
        else:
            input_stft = src_stft_mag
        output, (h_n, c_n) = self.rnn(input=input_stft)
        output = self.activation(self.output_proj(output))
        pred_mask = output.transpose(0, 1)
        return {
            "pred_mask": pred_mask
        }

@register_model("enh_transformer", dataclass=EnhTransformerConfig)
class EnhTransformer(BaseFairseqModel):
    def __init__(
        self, 
        cfg: EnhTransformerConfig,
    ):
        super().__init__()
        self.cfg = cfg
        assert cfg.setup in ['tiny', 'base', 'medium', 'large']
        if cfg.setup == 'tiny':
            d_model = 384
            nhead = 6
            dim_feedforward = 1536
            dropout = 0.1
            num_encoder_layers = 4
        elif cfg.setup == 'base':
            d_model = 512
            nhead = 8
            dim_feedforward = 2048
            dropout = 0.1
            num_encoder_layers = 6
        else:
            raise NotImplementedError

        if self.cfg.pos_enc == 'sinusoid':
            self.register_buffer("positional_embedding", sinusoids(3000, d_model))
        elif self.cfg.pos_enc == 'conv':
            self.pos_conv = make_conv_pos(
                d_model,
                128,
                16,
                is_batch_norm=False,
            )
        else:
            raise ValueError("Positional encoding undefined")
            
        self.input_proj = Linear(int(cfg.n_fft / 2 + 1), d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.output_proj = Linear(d_model, int(cfg.n_fft / 2 + 1))
        self.activation = {
            "relu": torch.nn.functional.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "none": torch.nn.Identity()
        }[cfg.activation]

    @classmethod
    def build_model(cls, cfg: EnhTransformerConfig, task: EnhancementTask):
        return EnhTransformer(cfg)

    def forward(self, src_stft_mag):
        assert len(src_stft_mag.size()) == 4
        src_stft_mag = (src_stft_mag[:, 0, :, :]).transpose(0, 1)
        if self.cfg.log1p:
            input_stft = torch.log1p(src_stft_mag)
        else:
            input_stft = src_stft_mag
        src = self.input_proj(input_stft)
        if self.cfg.pos_enc == "sinusoid":
            src = src + (self.positional_embedding[:src.size(0), :]).unsqueeze(1)
        elif self.cfg.pos_enc == "conv":
            src_conv = self.pos_conv(src.permute(1, 2, 0))
            src = src + src_conv.permute(2, 0, 1)
        else:
            raise ValueError("Positional encoding undefined")

        output = self.transformer_encoder(src=src)
        output = self.activation(self.output_proj(output))
        pred_mask = output.transpose(0, 1)
        return {
            "pred_mask": pred_mask
        }
