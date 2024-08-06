# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round
from pystoi.stoi import stoi
from pesq import pesq
import numpy as np

def si_sdr(reference, estimation):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf

    >>> np.random.seed(0)
    >>> reference = np.random.randn(100)
    >>> si_sdr(reference, reference)
    inf
    >>> si_sdr(reference, reference * 2)
    inf
    >>> si_sdr(reference, np.flip(reference))
    -25.127672346460717
    >>> si_sdr(reference, reference + np.flip(reference))
    0.481070445785553
    >>> si_sdr(reference, reference + 0.5)
    6.3704606032577304
    >>> si_sdr(reference, reference * 2 + 1)
    6.3704606032577304
    >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
    nan
    >>> si_sdr([reference, reference], [reference * 2 + 1, reference * 1 + 0.5])
    array([6.3704606, 6.3704606])

    """
    estimation, reference = np.broadcast_arrays(estimation, reference)

    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    # This is $\alpha$ after Equation (3) in [1].
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) \
        / reference_energy

    # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * reference

    # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)

@dataclass
class EnhCriterionConfig(FairseqDataclass):
    mask_type: str = field(
        default='PSM',
        metadata={"help": "mask type"},
    )
    loss_type: str = field(
        default='L1',
        metadata={"help": "loss type"},
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

@register_criterion("enh", dataclass=EnhCriterionConfig)
class EnhCriterion(FairseqCriterion):
    def __init__(
        self, cfg: EnhCriterionConfig, task: FairseqTask,
    ):
        super().__init__(task)
        self.cfg = cfg
        if self.cfg.loss_type == 'L1':
            self.loss = torch.nn.L1Loss(reduction='sum')
        elif self.cfg.loss_type == 'MSE':
            self.loss = torch.nn.MSELoss(reduction='sum')
        else:
            raise ValueError("Loss type not defined.")

    def forward(self, model, sample, reduce=True, **kwargs):
        #for k in sample.keys():
        #    if torch.is_tensor(sample[k]):
        #        print(k, sample[k].size())
        #    else:
        #        print(k, sample[k])

        #print(sample["src_stft_mag"].size())
        output_dict = model(sample["src_stft_mag"])
        pred_mask = output_dict["pred_mask"]
        #print("pred_mask", pred_mask.size())
        #print("src_stft_mag", sample["src_stft_mag"].size())
        pred_stft_mag = pred_mask.unsqueeze(1) * sample["src_stft_mag"]
        tgt_stft_mag = sample["tgt_stft_mag"]
        #print("tgt_stft_mag", tgt_stft_mag.size())

        if self.cfg.mask_type == "AM":
            ref_stft_mag = tgt_stft_mag
        elif self.cfg.mask_type == "PSM":
            ref_stft_mag = tgt_stft_mag * torch.cos(torch.angle(sample['src_stft_phase']) - torch.angle(sample['tgt_stft_phase']))
        elif self.cfg.mask_type == "NPSM":
            ref_stft_mag = tgt_stft_mag * F.relu(torch.cos(torch.angle(sample['src_stft_phase']) - torch.angle(sample['tgt_stft_phase'])))
        else:
            raise ValueError("Mask type not defined.")
        #print("ref_stft_mag", ref_stft_mag.size())
        loss = self.loss(pred_stft_mag, ref_stft_mag)
        #print("loss", loss.size())
        sample_size = pred_stft_mag.size(0) * pred_stft_mag.size(2)
        logging_output = {
            "loss": utils.item(loss.data), 
            "sample_size": sample_size,
            "nsentences": pred_stft_mag.size(0),
            "ntokens": sample_size,
        }
        #print("loss", loss)
        #print("sample_size", sample_size)
        #print("logging_output", logging_output)

        if not model.training:
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
            target_speech = sample["tgt_audios"][:, 0, :]
            assert len(pred_speech) == len(target_speech) == 1
            pred_speech, target_speech = pred_speech.data.cpu().numpy()[0, :], target_speech.data.cpu().numpy()[0, :]
            mixture = sample["src_audios"][0, :].data.cpu().numpy()
            pesq_score = pesq(16000, target_speech, pred_speech, 'wb')
            stoi_score = stoi(target_speech, pred_speech, 16000, extended=False)
            si_sdr_score = si_sdr(target_speech, pred_speech)
            si_sdr_score_mixture = si_sdr(target_speech, sample["src_audios"][0, :].data.cpu().numpy())
            si_sdri_score = si_sdr_score - si_sdr_score_mixture
            logging_output['pesq'] = pesq_score
            logging_output['stoi'] = stoi_score
            logging_output['si_sdri'] = si_sdri_score
            
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        pesq_sum = utils.item(sum(log.get("pesq", 0) for log in logging_outputs))
        stoi_sum = utils.item(sum(log.get("stoi", 0) for log in logging_outputs))
        si_sdri_sum = utils.item(sum(log.get("si_sdri", 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get("nsentences", 0) for log in logging_outputs))
        if pesq_sum > 0:
            metrics.log_scalar(
                "pesq", pesq_sum / nsentences, nsentences, round=3
            )
        if stoi_sum > 0:
            metrics.log_scalar(
                "stoi", stoi_sum / nsentences, nsentences, round=3
            )
        if si_sdri_sum != 0:
            metrics.log_scalar(
                "si_sdri", si_sdri_sum / nsentences, nsentences, round=3
            )
            

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
