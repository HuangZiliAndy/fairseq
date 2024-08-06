# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from fairseq.data import Dictionary, OnlineSeparationDatasetAMI, SeparationDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING
import torch
import soundfile as sf

from pystoi.stoi import stoi
from pesq import pesq
from fairseq.criterions.enh_criterion import si_sdr
from asteroid.metrics import get_metrics

logger = logging.getLogger(__name__)

@dataclass
class EnhancementConfig(FairseqDataclass):
    data_dir: str = field(default=MISSING, metadata={"help": "path to clean segments"})
    RIR_dir: str = field(default="none", metadata={"help": "path to RIR"})
    noise_dir: str = field(default="none", metadata={"help": "path to noise"})
    otf_data_simu: bool = field(
        default=True,
        metadata={"help": "On the fly data simulation"},
    )
    add_reverb: bool = field(
        default=True,
        metadata={"help": "whether to add reverberation"},
    )
    add_noise: bool = field(
        default=True,
        metadata={"help": "whether to add noise"},
    )
    noise_type: str = field(
        default="none",
        metadata={"help": "noise type"},
    )
    full_overlap: bool = field(
        default=False,
        metadata={"help": "whether to have full overlap"},
    )
    s1_first: bool = field(
        default=False,
        metadata={"help": "whether to put s1 first"},
    )
    s1_only: bool = field(
        default=False,
        metadata={"help": "whether to only have s1"},
    )
    crop_dur: float = field(
        default=20.0,
        metadata={"help": "crop the utterance longer than crop_dur"},
    )
    max_num_spk: int = field(
        default=1,
        metadata={"help": "max number of speakers"},
    )
    min_num_spk: int = field(
        default=1,
        metadata={"help": "max number of speakers"},
    )
    min_sir: float = field(
        default=5,
        metadata={"help": "min sir"},
    )
    max_sir: float = field(
        default=20,
        metadata={"help": "max sir"},
    )
    min_snr: float = field(
        default=5,
        metadata={"help": "min snr"},
    )
    max_snr: float = field(
        default=20,
        metadata={"help": "max snr"},
    )
    rate: int = field(
        default=16000,
        metadata={"help": "sample rate"},
    )
    n_fft: int = field(
        default=512,
        metadata={"help": "NFFT"},
    )
    hop_length: int = field(
        default=160,
        metadata={"help": "hop length"},
    )
    win_length: int = field(
        default=400,
        metadata={"help": "win length"},
    )
    chunk_size: float = field(
        default=4.0,
        metadata={"help": "chunk size"},
    )
    normalize: bool = field(
        default=True,
        metadata={"help": "whether to normalize"},
    )
    target: str = field(
        default='direct',
        metadata={"help": "training target"},
    )


@register_task("enhancement", dataclass=EnhancementConfig)
class EnhancementTask(FairseqTask):

    def __init__(
        self,
        cfg: EnhancementConfig,
    ) -> None:
        super().__init__(cfg)

        self.cfg = cfg

    @classmethod
    def setup_task(
        cls, cfg: EnhancementConfig, **kwargs
    ) -> "EnhancementConfig":
        return cls(cfg)

    def load_dataset(self, split: str, **kwargs) -> None:
        if self.cfg.otf_data_simu:
            data_dir = "{}/{}_ihm_clean_filter".format(self.cfg.data_dir, split)
        else:
            data_dir = "{}/{}".format(self.cfg.data_dir, split)
    
        RIR_dir = "{}/{}".format(self.cfg.RIR_dir, split)
        if 'train' in split:
            noise_split = 'tr'
            chunk_size = self.cfg.chunk_size 
            shuffle = True
        elif 'dev' in split:
            noise_split = 'cv'
            chunk_size = -1 # use entire sentence for validation
            shuffle = False
        elif 'test' in split:
            noise_split = 'tt'
            chunk_size = -1 # use entire sentence for validation
            shuffle = False
        else:
            raise ValueError("Split undefined.")
        noise_scp_file = '{}/{}.scp'.format(self.cfg.noise_dir, noise_split)

        if self.cfg.otf_data_simu:
            self.datasets[split] = OnlineSeparationDatasetAMI(
                data_dir,
                add_reverb=self.cfg.add_reverb,
                RIR_dir=RIR_dir,
                add_noise=self.cfg.add_noise,
                noise_type=self.cfg.noise_type,
                noise_scp_file=noise_scp_file,
                full_overlap=self.cfg.full_overlap,
                s1_first=self.cfg.s1_first,
                s1_only=self.cfg.s1_only,
                crop_dur=self.cfg.crop_dur,
                max_num_spk=self.cfg.max_num_spk,
                min_num_spk=self.cfg.min_num_spk,
                min_sir=self.cfg.min_sir,
                max_sir=self.cfg.max_sir,
                min_snr=self.cfg.min_snr,
                max_snr=self.cfg.max_snr,
                rate=self.cfg.rate,
                channel='0',
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                win_length=self.cfg.win_length,
                chunk_size=chunk_size,
                normalize=self.cfg.normalize,
                target=self.cfg.target,
                ref_channel=0,
                early_rir_dur=0.05,
            )
        else:
            if self.cfg.target == 'direct':
                tgt_conds = 's1_direct'
            elif self.cfg.target == 'clean':
                tgt_conds = 's1'
            elif self.cfg.target == 'none':
                tgt_conds = 'none'
            else:
                raise ValueError("Invalid target condition")
            self.datasets[split] = SeparationDataset(
                data_dir,
                rate=self.cfg.rate,
                src_cond='wav',
                tgt_conds=tgt_conds,
                channel='0',
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                win_length=self.cfg.win_length,
                chunk_size=chunk_size,
                ref_channel=0,
                shuffle=shuffle,
            )

    def inference_step(
        self, models, sample, infer_cfg
    ): 
        pred_dir = "{}/predictions".format(infer_cfg.common_eval.results_path)
        models[0].eval()
        with torch.no_grad():
            pred_mask = models[0](sample["src_stft_mag"])["pred_mask"]
        pred_stft_mag = pred_mask.unsqueeze(1) * sample["src_stft_mag"]
        pred_stft = pred_stft_mag * sample['src_stft_phase']
        pred_stft = (pred_stft[:, 0, :, :]).transpose(1, 2)
        pred_speech = torch.istft(
            pred_stft, 
            n_fft=self.cfg.n_fft, 
            hop_length=self.cfg.hop_length, 
            win_length=self.cfg.win_length, 
            window=torch.hann_window(self.cfg.win_length).to(pred_stft.device), 
            return_complex=False,
            length=sample["src_audios"].size(1)
        )
        assert len(pred_speech) == 1
        src_speech = sample["src_audios"][0, :].data.cpu().numpy()
        pred_speech = pred_speech[0, :].data.cpu().numpy()
        
        if sample["tgt_audios"] is None:
            target_speech = None
        else:
            target_speech = sample["tgt_audios"][0, 0, :].data.cpu().numpy()

        if infer_cfg.save_audio:
            #sf.write('{}/{}_src.wav'.format(pred_dir, sample["uttname_list"][0]), src_speech, 16000)
            #sf.write('{}/{}_tgt.wav'.format(pred_dir, sample["uttname_list"][0]), target_speech, 16000)
            sf.write('{}/{}_pred.wav'.format(pred_dir, sample["uttname_list"][0]), pred_speech, 16000)

        if target_speech is not None:
            COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi", "pesq"]
            utt_metrics = get_metrics(
                src_speech,
                target_speech,
                pred_speech,
                sample_rate=16000,
                metrics_list=COMPUTE_METRICS,
            )
        else:
            utt_metrics = {}

        #pesq_score = pesq(16000, target_speech, pred_speech, 'wb')
        #stoi_score = stoi(target_speech, pred_speech, 16000, extended=False)
        #si_sdr_score = si_sdr(target_speech, pred_speech)
        #si_sdr_score_mixture = si_sdr(target_speech, sample["src_audios"][0, :].data.cpu().numpy())
        #si_sdri_score = si_sdr_score - si_sdr_score_mixture
        #metric_dict = {
        #    "pesq": pesq_score,
        #    "stoi": stoi_score,
        #    "si_sdri": si_sdri_score,
        #    "si_sdr": si_sdr_score,
        #}
        return utt_metrics
            
            
