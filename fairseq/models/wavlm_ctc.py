# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
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
from asteroid import DPTNet
from fairseq.models.enh_models import EnhTransformer

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
    enh_model_type: str = field(
        default="none", metadata={"help": "type of enhancement model"}
    )
    enh_model_path: str = field(
        default="none", metadata={"help": "enhancement model path"}
    )
    freeze_enh_model: str = field(
        default=True, metadata={"help": "whether to freeze enhancement model"}
    )
    norm_enh_output: int = field(
        default=0, metadata={"help": "how to normalize enhancement output"}
    )
    asr_model_path: str = field(
        default="none", metadata={"help": "asr model path"}
    )

class DictToAttr:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DictToAttr(value)
            self.__dict__[key] = value

@register_model("wavlm_ctc", dataclass=WavlmAsrConfig)
class WavlmCtc(BaseFairseqModel):
    def __init__(
        self, 
        cfg: WavlmAsrConfig,
        vocab_size: int,
    ):
        super().__init__()
        self.cfg = cfg
        if cfg.enh_model_type == "none":
            self.enh_model = None
            print("Not using speech enhancement model")
        elif cfg.enh_model_type == "DPTNet":
            if cfg.enh_model_path == "none":
                self.enh_model = DPTNet()
            else:
                self.enh_model = DPTNet.from_pretrained(cfg.enh_model_path)
                print("Using pretrained speech enhancement model {}".format(cfg.enh_model_path))
        elif cfg.enh_model_type == "EnhTransformer":
            if cfg.enh_model_path == "none":
                raise ValueError("empty enhancement path")
            else:
                enh_model = torch.load(cfg.enh_model_path)
                self.enh_model_cfg = enh_model['cfg']
                self.enh_model = EnhTransformer(DictToAttr(enh_model['cfg']['model']))
                self.enh_model.load_state_dict(enh_model['model'])
                print("Using pretrained speech enhancement model {}".format(cfg.enh_model_path))
        else:
            raise ValueError("Enhancement model undefined.")
        if self.enh_model is not None and cfg.freeze_enh_model:
            for name, param in self.enh_model.named_parameters():
                param.requires_grad = False

        self.wavlm_model = WavLMModel.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.cache_dir,
            vocab_size=vocab_size,
        )
        self.wavlm_model.freeze_feature_encoder()
        self.model = WavLMForCTC(self.wavlm_model.config)
        self.model.wavlm = self.wavlm_model

        if cfg.asr_model_path != "none":
            assert os.path.exists(cfg.asr_model_path)
            asr_model_state_dict = torch.load(cfg.asr_model_path)["model"]
            missing_keys, unexpected_keys = self.load_state_dict(asr_model_state_dict, strict=False)
            assert len(unexpected_keys) == 0

        target_modules = cfg.target_modules.split(',')

        if cfg.use_lora:
            peft_config = LoraConfig(
                target_modules=target_modules, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            )
            self.model = get_peft_model(self.model, peft_config)

        for name, param in self.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True
            if not cfg.freeze_layer_norm:
                if "layer_norm" in name or "norm" in name:
                    param.requires_grad = True

        # Print trainable and frozen parameters
        trainable_params, frozen_params = [], []
        
        for name, param in self.named_parameters():
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

        self.norm_enh_output = cfg.norm_enh_output

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

    def is_model_fp16(self, model):
        first_param = next(model.parameters(), None)
        if first_param is not None and first_param.dtype == torch.float16:
            return True
        return False

    def forward(self, source, padding_mask):
        if self.enh_model is not None:
            if self.cfg.enh_model_type == "DPTNet":
                enh_source = (self.enh_model(source)).squeeze(1)
            elif self.cfg.enh_model_type == "EnhTransformer":
                task_cfg = DictToAttr(self.enh_model_cfg['task'])
                src_stft = torch.stft(
                    source,
                    task_cfg.n_fft,
                    hop_length=task_cfg.hop_length,
                    win_length=task_cfg.win_length,
                    window=torch.hann_window(task_cfg.win_length).to(source.device),
                    return_complex=True
                )
                src_stft = (src_stft.view(src_stft.size(0), -1, src_stft.size(1), src_stft.size(2))).permute(0,1,3,2)
                src_stft_mag = torch.abs(src_stft) + 1e-10
                src_stft_phase = src_stft / src_stft_mag
                if self.is_model_fp16(self.enh_model):
                    src_stft_mag = src_stft_mag.half()
                pred_mask = self.enh_model(src_stft_mag)["pred_mask"]
                pred_stft = src_stft_mag[:, 0, :, :] * pred_mask * src_stft_phase[:, 0, :, :]
                enh_source = torch.istft(
                    pred_stft.transpose(1, 2), 
                    n_fft=task_cfg.n_fft, 
                    hop_length=task_cfg.hop_length, 
                    win_length=task_cfg.win_length, 
                    window=torch.hann_window(task_cfg.win_length).to(pred_stft.device), 
                    return_complex=False,
                    length=source.size(1),
                )
                if self.is_model_fp16(self.enh_model):
                    enh_source = enh_source.half()
            else:
                raise ValueError("Enhancement model undefined.")

            if self.norm_enh_output == 0:
                source = enh_source
            elif self.norm_enh_output == 1: 
                origin_max = torch.max(torch.abs(source), 1, keepdim=True)[0]
                enh_max = (torch.max(torch.abs(enh_source), 1, keepdim=True)[0]) + 1e-10
                enh_source = enh_source / enh_max * origin_max
                source = enh_source
            elif self.norm_enh_output == 2:
                enh_max = (torch.max(torch.abs(enh_source), 1, keepdim=True)[0]) + 1e-10
                enh_source = enh_source / enh_max
                source = enh_source
            else:
                raise ValueError("Norm method not defined.")

        output = self.model(input_values=source, attention_mask=~padding_mask, return_dict=True)
        return {
            "encoder_out": output['logits'].transpose(0, 1),  # T x B x C
            "encoder_padding_mask": ~output['attention_mask'],  # B x T
            "padding_mask": ~output['attention_mask'],
        }
