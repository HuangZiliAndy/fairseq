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
from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingConfig,
    HubertPretrainingTask,
)
from fairseq import utils
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

@dataclass
class WavlmAsrConfig(FairseqDataclass):
    model_name: str = field(
        default="microsoft/wavlm-base-plus", metadata={"help": "model name"}
    )
    cache_dir: str = field(
        default="/scratch4/lgarci27/hzili1/workspace/huggingface_models", metadata={"help": "cache directory"}
    )
    freeze_feature_encoder: bool = field(
        default=False,
        metadata={"help": "whether to freeze the feature encoder"},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "whether to use LoRA"},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank"},
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"},
    )
    target_modules: str = field(
        default="q_proj,v_proj",
        metadata={"help": "Target modules"},
    )
    freeze_layer_norm: bool = field(
        default=False,
        metadata={"help": "whether to freeze the layer norm"},
    )

    

@register_model("wavlm_ctc", dataclass=WavlmAsrConfig)
class WavlmCtc(BaseFairseqModel):
    def __init__(
        self, 
        cfg: WavlmAsrConfig,
        vocab_size: int,
    ):
        super().__init__()
        self.wavlm_model = WavLMModel.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.cache_dir,
            vocab_size=vocab_size,
        )
        self.wavlm_model.freeze_feature_encoder()
        self.model = WavLMForCTC(self.wavlm_model.config)
        self.model.wavlm = self.wavlm_model

        target_modules = cfg.target_modules.split(',')

        if cfg.use_lora:
            peft_config = LoraConfig(
                target_modules=target_modules, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            )
            self.model = get_peft_model(self.model, peft_config)

        for name, param in self.model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True
            if not cfg.freeze_layer_norm and "layer_norm" in name:
                param.requires_grad = True

        # Print trainable and frozen parameters
        trainable_params, frozen_params = [], []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)
        
        print("Trainable parameters:")
        for name in trainable_params:
            print(name)
        print('-' * 80)
        print("\nFrozen parameters:")
        for name in frozen_params:
            print(name)

    @classmethod
    def build_model(cls, cfg: WavlmAsrConfig, task: HubertPretrainingTask):
        return WavlmCtc(cfg, len(task.target_dictionary))

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, source, padding_mask):
        output = self.model(input_values=source, attention_mask=~padding_mask, return_dict=True)
        return {
            "encoder_out": output['logits'].transpose(0, 1),  # T x B x C
            "encoder_padding_mask": ~output['attention_mask'],  # B x T
            "padding_mask": ~output['attention_mask'],
        }
