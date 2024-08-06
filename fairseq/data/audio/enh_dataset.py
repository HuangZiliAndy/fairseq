# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ LibriMix speech separation dataset ]
#   Author       [ Zili Huang ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import sys
import random
import numpy as np
from scipy import signal

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

import librosa
import soundfile as sf
from torch.utils.data import DataLoader
#from s3prl.downstream.sep_ami.noise import * 
from fairseq.data.fairseq_dataset import FairseqDataset

def compute_snr(signal, noise):
    sig_pwr = np.mean(signal**2)
    noz_pwr = np.mean(noise**2)
    if sig_pwr == 0.0:
        return -np.inf
    elif noz_pwr == 0.0:
        return np.inf
    else:
        return 10.0 * np.log10(sig_pwr / noz_pwr)

# Code from https://github.com/fgnt/sms_wsj
def get_rir_start_sample(h, level_ratio=1e-1):
    """Finds start sample in a room impulse response.

    Selects that index as start sample where the first time
    a value larger than `level_ratio * max_abs_value`
    occurs.

    If you intend to use this heuristic, test it on simulated and real RIR
    first. This heuristic is developed on MIRD database RIRs and on some
    simulated RIRs but may not be appropriate for your database.

    If you want to use it to shorten impulse responses, keep the initial part
    of the room impulse response intact and just set the tail to zero.

    Params:
        h: Room impulse response with Shape (num_samples,)
        level_ratio: Ratio between start value and max value.

    >>> get_rir_start_sample(np.array([0, 0, 1, 0.5, 0.1]))
    2
    """
    assert level_ratio < 1, level_ratio
    if h.ndim > 1:
        assert h.shape[0] < 20, h.shape
        h = np.reshape(h, (-1, h.shape[-1]))
        start_sample_list = [get_rir_start_sample(h_, level_ratio=level_ratio) for h_ in h]
        return np.min(start_sample_list)

    abs_h = np.abs(h)
    max_index = np.argmax(abs_h)
    max_abs_value = abs_h[max_index]
    # +1 because python excludes the last value
    larger_than_threshold = abs_h[:max_index + 1] > level_ratio * max_abs_value

    # Finds first occurrence of max
    rir_start_sample = np.argmax(larger_than_threshold)
    return rir_start_sample

class SeparationDataset(FairseqDataset):
    def __init__(
        self,
        data_dir,
        rate=16000,
        src_cond='wav',
        tgt_conds='s1_direct,s2_direct',
        channel='0',
        n_fft=512,
        hop_length=160,
        win_length=400,
        chunk_size=4.0,
        ref_channel=0,
        shuffle=False,
    ):
        super(SeparationDataset, self).__init__()
        self.data_dir = data_dir
        self.rate = rate
        self.src_cond = src_cond
        if tgt_conds == 'none':
            self.tgt_conds = [] 
        else:
            self.tgt_conds = [cond for cond in tgt_conds.split(',')]
        self.channel = [int(c) for c in channel.split(',')]
        if len(self.channel) == 1:
            self.channel = self.channel[0]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_srcs = len(self.tgt_conds)

        for f in [self.src_cond] + self.tgt_conds: 
            print(f)
            assert os.path.exists("{}/{}.scp".format(data_dir, f))
        self.utt2srcpath = self.get_utt2path("{}/{}.scp".format(data_dir, src_cond))
        if len(self.tgt_conds) == 0:
            self.utt2tgtpaths = [] 
        else:
            self.utt2tgtpaths = [self.get_utt2path("{}/{}.scp".format(data_dir, tgt_cond)) for tgt_cond in self.tgt_conds]
        self.uttlist = list(self.utt2srcpath.keys())
        self.uttlist.sort()
        self.chunk_size = int(chunk_size * self.rate)
        self.ref_channel = ref_channel
        self.shuffle = shuffle

    def get_utt2path(self, fname):
        utt2path = {}
        with open(fname, 'r') as fh:
            content = fh.readlines()
        for line in content:
            line = line.strip('\n')
            utt, path = line.split()
            utt2path[utt] = path
        return utt2path

    def __len__(self):
        return len(self.uttlist)

    def __getitem__(self, i):
        uttname = self.uttlist[i]

        src_audio_path = self.utt2srcpath[uttname]
        src_audio, sr = sf.read(src_audio_path)
        if len(src_audio.shape) == 2:
            src_audio = src_audio[:, self.channel]
        elif len(src_audio.shape) == 1:
            assert self.channel == 0
        else:
            raise ValueError("Invalid audio shape")

        if len(self.tgt_conds) == 0:
            tgt_audio = None
        else:
            tgt_list = []
            for j, tgt_cond in enumerate(self.tgt_conds):
                tgt_audio_path = self.utt2tgtpaths[j][uttname]
                tgt_audio, sr = sf.read(tgt_audio_path)
                if len(tgt_audio.shape) == 2:
                    tgt_audio = tgt_audio[:, self.ref_channel]
                tgt_list.append(tgt_audio)
            tgt_audio = np.stack(tgt_list, 0)

        if tgt_audio is not None:
            assert src_audio.shape[0] == tgt_audio.shape[1]
        num_samples = src_audio.shape[0]

        if self.chunk_size > 0:
            if self.chunk_size < num_samples:
                start_sample = np.random.randint(low=0, high=num_samples-self.chunk_size)
                if tgt_audio is not None:
                    tgt_audio = tgt_audio[:, start_sample:start_sample+self.chunk_size]
                src_audio = src_audio[start_sample:start_sample+self.chunk_size]
            else:
                if tgt_audio is not None:
                    tgt_audio = np.pad(tgt_audio, [(0, 0), (0, self.chunk_size - num_samples)])
                if len(src_audio.shape) == 2:
                    src_audio = np.pad(src_audio, [(0, self.chunk_size - num_samples), (0, 0)])
                elif len(src_audio.shape) == 1:
                    src_audio = np.pad(src_audio, (0, self.chunk_size - num_samples))

        src_audio = np.transpose(src_audio)
        return src_audio, tgt_audio, uttname

    def collater(self, batch):
        sorted_batch = sorted(batch, key=lambda x: -x[0].shape[0])
        bs = len(sorted_batch)
        src_audios = torch.stack([torch.from_numpy(sorted_batch[i][0]).float() for i in range(bs)], 0)
        if sorted_batch[0][1] is None:
            tgt_audios = None
        else:
            tgt_audios = torch.stack([torch.from_numpy(sorted_batch[i][1]).float() for i in range(bs)], 0)
        uttname_list = [sorted_batch[i][2] for i in range(bs)]
        src_stft = torch.stft(
                src_audios.view(-1, src_audios.size(-1)),
                self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=torch.hann_window(self.win_length).to(src_audios.device),
                return_complex=True
            )
        src_stft = (src_stft.view(bs, -1, src_stft.size(1), src_stft.size(2))).permute(0,1,3,2)
        src_stft_mag = torch.abs(src_stft) + 1e-10
        src_stft_phase = src_stft / src_stft_mag
        
        if tgt_audios is None:
            tgt_stft_mag = None
            tgt_stft_phase = None
        else:
            tgt_stft = torch.stft(
                    tgt_audios.view(-1, tgt_audios.size(-1)),
                    self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=torch.hann_window(self.win_length).to(tgt_audios.device),
                    return_complex=True
                )
            tgt_stft = (tgt_stft.view(bs, -1, tgt_stft.size(1), tgt_stft.size(2))).permute(0,1,3,2)
            tgt_stft_mag = torch.abs(tgt_stft) + 1e-10
            tgt_stft_phase = tgt_stft / tgt_stft_mag

        batch = {
            "src_audios": src_audios,
            "tgt_audios": tgt_audios,
            "src_stft_mag": src_stft_mag,
            "tgt_stft_mag": tgt_stft_mag,
            "src_stft_phase": src_stft_phase,
            "tgt_stft_phase": tgt_stft_phase,
            "uttname_list": uttname_list,
        }
        return batch

    def size(self, index):
        if self.chunk_size > 0:
            return self.chunk_size
        else:
            uttname = self.uttlist[index]
            audio_path = self.utt2srcpath[uttname]
            return sf.info(audio_path).frames

    def num_tokens(self, index):
        return self.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            return np.random.permutation(len(self))
        else:
            return np.arange(len(self))

class SeparationDatasetUttGroup(Dataset):
    def __init__(
        self,
        data_dir,
        rate=16000,
        channel='0',
        normalize=0,
    ):
        super(SeparationDatasetUttGroup, self).__init__()
        for f in ['wav.scp', 'text']:
            assert os.path.exists("{}/{}".format(data_dir, f))
        self.utt2path = self.get_utt2path("{}/wav.scp".format(data_dir)) 
        self.utt2textlist = self.get_utt2textlist("{}/text".format(data_dir)) 
        self.uttlist = list(self.utt2path.keys())
        self.uttlist.sort()
        self.channel = [int(c) for c in channel.split(',')]
        self.normalize = normalize

    def get_utt2path(self, fname):
        utt2path = {}
        with open(fname, 'r') as fh:
            content = fh.readlines()
        for line in content:
            line = line.strip('\n')
            utt, path = line.split()
            utt2path[utt] = path
        return utt2path

    def get_utt2textlist(self, fname):
        utt2textlist = {}
        with open(fname, 'r') as fh:
            content = fh.readlines()
        for line in content:
            line = line.strip('\n')
            utt, text = line.split(None, 1)
            textlist = text.split('#')
            utt2textlist[utt] = [' '.join(text.split()[1:]) for text in textlist]
        return utt2textlist

    def __getitem__(self, i):
        uttname = self.uttlist[i]
        uttpath = self.utt2path[uttname]
        audio, sr = sf.read(uttpath)
        if self.normalize:
            audio = audio / np.max(np.abs(audio))
        if len(audio.shape) == 2:
            audio = audio[:, self.channel]
        elif len(audio.shape) == 1:
            assert self.channel == 0
            audio = np.expand_dims(audio, axis=1)
        else:
            raise ValueError("Invalid audio shape")
        textlist = self.utt2textlist[uttname]
        assert len(audio.shape) == 2
        audio = np.transpose(audio)
        return audio, textlist, uttname

    def __len__(self):
        return len(self.uttlist)

class OnlineSeparationDatasetAMI(FairseqDataset):
    def __init__(
        self,
        data_dir,
        add_reverb=True,
        RIR_dir=None,
        add_noise=False,
        noise_type='none',
        noise_scp_file=None,
        full_overlap=False,
        s1_first=False,
        s1_only=False,
        crop_dur=10.0,
        max_num_spk=2,
        min_num_spk=1,
        min_sir=-5,
        max_sir=5,
        min_snr=5,
        max_snr=20,
        rate=16000,
        channel='0',
        n_fft=512,
        hop_length=160,
        win_length=400,
        chunk_size=4.0,
        normalize=True,
        target='clean',
        ref_channel=0,
        early_rir_dur=0.05,
    ):
        super(OnlineSeparationDatasetAMI, self).__init__()
        self.data_dir = data_dir
        self.utt2path = self.get_utt2path("{}/wav.scp".format(data_dir))
        self.utt2path, self.reco2dur, self.location2dur, self.location2spk, self.spk2dur, self.spk2utt = self.parse_data_dir(data_dir)
        self.uttlist = list(self.utt2path.keys())
        self.uttlist.sort()
        self.spklist = list(self.spk2dur.keys())
        self.spklist.sort()
        self.spkprob = np.array([self.spk2dur[spk] for spk in self.spklist])
        self.spkprob = self.spkprob / np.sum(self.spkprob)
        self.add_reverb, self.add_noise = add_reverb, add_noise
        self.noise_type = noise_type
        self.ref_channel = ref_channel
        self.target = target

        if self.add_reverb:
            self.RIR_list = self.get_RIRlist(RIR_dir)
        else:
            self.RIR_list = None

        if self.add_noise:
            self.noise2path = self.get_utt2path(noise_scp_file)
            self.noiselist = list(self.noise2path.keys())
            self.noiselist.sort()
        else:
            self.noise2path = None
            self.noiselist = None
        print("Loading {} clean audios, {} speakers".format(len(self.uttlist), len(self.spklist)))
        if self.add_reverb:
            print("{} RIR files".format(len(self.RIR_list)))
        if self.add_noise:
            print("{} noise files".format(len(self.noiselist)))

        
        # parameters for mixture simulation
        self.full_overlap = full_overlap
        self.s1_first = s1_first
        self.s1_only = s1_only

        self.crop_dur = crop_dur
        self.num_spk_range = [min_num_spk, max_num_spk]
        self.sir_range = [min_sir, max_sir]
        self.snr_range = [min_snr, max_snr]

        self.rate = rate
        self.channel = [int(c) for c in channel.split(',')]
        if len(self.channel) == 1:
            self.channel = self.channel[0]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.chunk_size = int(chunk_size * self.rate)
        self.normalize = normalize
        print("Noise type {}".format(self.noise_type))

        self.early_rir_samples = int(early_rir_dur * self.rate)

    def get_utt2path(self, fname):
        utt2path = {}
        with open(fname, 'r') as fh:
            content = fh.readlines()
        for line in content:
            line = line.strip('\n')
            utt, path = line.split()
            utt2path[utt] = path
        return utt2path

    def get_reco2dur(self, fname):
        reco2dur = {}
        with open(fname, 'r') as fh:
            content = fh.readlines()
        for line in content:
            line = line.strip('\n')
            utt, duration = line.split()
            reco2dur[utt] = float(duration)
        return reco2dur

    def get_RIRlist(self, RIRdir):
        RIRlist = ["{}/{}".format(RIRdir, f) for f in os.listdir(RIRdir)]
        RIRlist.sort()
        return RIRlist

    def parse_data_dir(self, src_dir):
        assert os.path.exists("{}/wav.scp".format(src_dir)) and os.path.exists("{}/reco2dur".format(src_dir))
        utt2path = self.get_utt2path("{}/wav.scp".format(src_dir))
        reco2dur = self.get_reco2dur("{}/reco2dur".format(src_dir))
        uttlist = list(utt2path.keys())
        uttlist.sort()
        location2dur, location2spk, spk2dur, spk2utt = {}, {}, {}, {}
        for utt in uttlist:
            location = utt[0]
            spk = utt.split('_')[1]
            duration = reco2dur[utt]
            if location not in location2dur:
                location2dur[location] = 0
            if location not in location2spk:
                location2spk[location] = []
            if spk not in spk2dur:
                spk2dur[spk] = 0
            if spk not in spk2utt:
                spk2utt[spk] = []
            location2dur[location] += duration
            location2spk[location].append(spk)
            spk2dur[spk] += duration
            spk2utt[spk].append(utt)
        for loc in location2spk.keys():
            spklist = list(set(location2spk[loc]))
            spklist.sort()
            location2spk[loc] = spklist
        return utt2path, reco2dur, location2dur, location2spk, spk2dur, spk2utt 

    def compute_energy_dB(self, src):
        #assert len(src.shape) == 1
        return 10 * np.log10(max(1e-20, np.mean(src ** 2)))
    
    def scale_audios(self, srcs, snr_list):
        energy_dB_list = [self.compute_energy_dB(src) for src in srcs]
        gain_list = [min(40, -snr_list[i]+energy_dB_list[0]-energy_dB_list[i]) for i in range(len(energy_dB_list))]
        scale_srcs = [srcs[i] * np.power(10, (gain_list[i] / 20.)) for i in range(len(srcs))]
        #energy_dB_list = [self.compute_energy_dB(src) for src in scale_srcs]
        return scale_srcs, gain_list

    def align_audios(self, clean_srcs_list, add_noise, full_overlap, s1_first):
        new_clean_srcs_list = []
        num_samples = [src.shape[0] for src in clean_srcs_list]
        if add_noise:
            num_speech_srcs = len(clean_srcs_list) - 1
            noise = clean_srcs_list[-1]
        else:
            num_speech_srcs = len(clean_srcs_list)
            noise = None
    
        if num_speech_srcs == 1:
            total_sample = num_samples[0]
            new_clean_srcs_list.append(clean_srcs_list[0])
            start_sample = [0]
        elif num_speech_srcs == 2:
            if full_overlap:
                start_s1, start_s2 = 0, 0
            else:
                if s1_first:
                    start_s1 = 0
                    start_s2 = np.random.randint(low=0, high=num_samples[0])
                else:
                    if np.random.randint(2):
                        start_s1 = 0
                        start_s2 = np.random.randint(low=0, high=num_samples[0])
                    else:
                        start_s1 = np.random.randint(low=0, high=num_samples[1])
                        start_s2 = 0
            total_sample = max(start_s1 + num_samples[0], start_s2 + num_samples[1])
            s1, s2 = np.zeros(total_sample,), np.zeros(total_sample,)
            s1[start_s1:start_s1 + num_samples[0]] = clean_srcs_list[0]
            s2[start_s2:start_s2 + num_samples[1]] = clean_srcs_list[1]
            new_clean_srcs_list.append(s1)
            new_clean_srcs_list.append(s2)
            start_sample = [start_s1, start_s2]
    
        if add_noise:
            if noise.shape[0] <= total_sample:
                noise = np.repeat(noise, int(total_sample / noise.shape[0]) + 1)
            start_noise = np.random.randint(low=0, high=noise.shape[0] - total_sample)
            new_clean_srcs_list.append(noise[start_noise:start_noise + total_sample])
            start_sample.append(0)
        assert len(new_clean_srcs_list) == len(start_sample)
    
        length_list = [src.shape[0] for src in new_clean_srcs_list]
        assert np.min(length_list) == np.max(length_list)
        return new_clean_srcs_list, start_sample 

    def mch_rir_conv(self, input_wav, mch_rir, early_rir_samples):
        input_wav = np.expand_dims(input_wav, axis=0)
        start_idx = np.argmax(mch_rir[0])
        #start_idx = get_rir_start_sample(mch_rir)
        end_idx_direct = min(start_idx + early_rir_samples, mch_rir.shape[1])
    
        mch_rir_early = mch_rir.copy()
        mch_rir_early[:, end_idx_direct:] = 0
    
        #start_idx = np.argmax(mch_rir[0])
        reverb_wav = signal.fftconvolve(input_wav, mch_rir, mode="full")
        reverb_wav = reverb_wav[:, start_idx:start_idx + input_wav.shape[-1]]
        direct_wav = signal.fftconvolve(input_wav, mch_rir_early, mode="full")
        direct_wav = direct_wav[:, start_idx:start_idx + input_wav.shape[-1]]
        return reverb_wav, direct_wav

    def __len__(self):
        return len(self.uttlist)

    def __getitem__(self, idx):
        #print('-' * 80)
        #print("self.num_spk_range", self.num_spk_range)
        num_spk = np.random.randint(self.num_spk_range[0], self.num_spk_range[1]+1)
        #print("num_spk", num_spk)
        spk_list = np.random.choice(self.spklist, size=num_spk, replace=True, p=self.spkprob)
        #print("spk_list", spk_list)
        uttname_list = []
        for spk in spk_list:
            utt_dur_list = [self.reco2dur[utt] for utt in self.spk2utt[spk]]
            utt_prob = np.array(utt_dur_list) / np.sum(utt_dur_list)
            utt = np.random.choice(self.spk2utt[spk], replace=True, p=utt_prob)
            uttname_list.append(utt)

        clean_srcs_list, snr_list = [], []
        for i, utt in enumerate(uttname_list):
            audio, _ = sf.read(self.utt2path[utt])
            duration = audio.shape[0] / 16000.0
            if duration <= self.crop_dur:
                clean_srcs_list.append(audio)
                snr_list.append(0 if i == 0 else np.random.uniform(self.sir_range[0], self.sir_range[1]))
            else:
                crop_len = np.random.uniform(low=3.0, high=10.0)
                crop_len = round(crop_len, 2)
                crop_sample = int(16000.0 * crop_len)
                start_sample = np.random.randint(low=0, high=audio.shape[0]-crop_sample)
                clean_srcs_list.append(audio[start_sample:start_sample + crop_sample])
                snr_list.append(0 if i == 0 else np.random.uniform(self.sir_range[0], self.sir_range[1]))
        #print("clean_srcs_list", [src.shape for src in clean_srcs_list])
        #print("snr_list", snr_list)

        if self.add_noise:
            # randomly select noise file
            noise_utt = np.random.choice(self.noiselist)
            noise_file = self.noise2path[noise_utt]
            noise, sr = sf.read(noise_file)
            assert sr == 16000
            if len(noise.shape) == 1:
                noise_channel = 0
            elif len(noise.shape) == 2:
                noise_channel = int(np.random.randint(noise.shape[1]))
                noise = noise[:, noise_channel]
            noise_dur = noise.shape[0] / 16000.0
            noise_crop_len = 20.0
            if noise_dur > noise_crop_len:
                crop_sample = int(16000.0 * noise_crop_len)
                start_sample = np.random.randint(low=0, high=noise.shape[0]-crop_sample)
                noise = noise[start_sample:start_sample+crop_sample]
            clean_srcs_list.append(noise)
            snr_list.append(np.random.uniform(self.snr_range[0], self.snr_range[1]))
            uttname_list.append(noise_utt)
        assert len(clean_srcs_list) == len(snr_list)
        #print("clean_srcs_list", [src.shape for src in clean_srcs_list])
        #print("snr_list", snr_list)
        #print("uttname_list", uttname_list)

        ## Scale the audios according to the SNR
        #clean_srcs_list, gain_list = self.scale_audios(clean_srcs_list, snr_list)

        # Decide the start time of each clean segments
        clean_srcs_list_align, start_sample = self.align_audios(clean_srcs_list, add_noise=self.add_noise, full_overlap=self.full_overlap, s1_first=self.s1_first)

        if self.s1_only:
            start_sample_s1, end_sample_s1 = start_sample[0], start_sample[0] + clean_srcs_list[0].shape[0] 
            clean_srcs_list_align = [s[start_sample_s1:end_sample_s1] for s in clean_srcs_list_align]

        #clean_srcs = np.stack(clean_srcs_list_align, 0)
        #duration = clean_srcs_list_align[0].shape[0] / 16000.0

        if self.add_reverb:
            RIR_file = np.random.choice(self.RIR_list)
            RIR_dict = np.load(RIR_file)
            RIR = RIR_dict['rir']
            channel_id = random.randint(0, RIR.shape[0] - 1)
            RIR = RIR[channel_id:channel_id+1, :, :]
            #print("RIR", RIR.shape)

            reverb_srcs_list, direct_srcs_list = [], []
            for i in range(num_spk):
                reverb_src, direct_src = self.mch_rir_conv(clean_srcs_list_align[i], RIR[:, i, :], self.early_rir_samples)
                reverb_srcs_list.append(reverb_src)
                direct_srcs_list.append(direct_src)

            if self.add_noise:
                if self.noise_type == 'diffuse':
                    diffuse = DiffuseNoise(snr=0, signal=clean_srcs_list_align[-1])
                    noise = diffuse.generate_noise(RIR_dict['mic_pos'].T, 16000)
                    reverb_srcs_list.append(noise)
                    #direct_srcs_list.append(noise)
                elif self.noise_type == 'white':
                    noise = np.random.randn(info_dict['mic_pos'].shape[0], clean_srcs_list_align[-1].shape[0])
                    reverb_srcs_list.append(noise)
                    #direct_srcs_list.append(noise)
                elif self.noise_type == 'point':
                    reverb_noise, direct_noise = mch_rir_conv(clean_srcs_list_align[num_spk], rir[:, num_spk, :], early_rir_samples)
                    reverb_srcs_list.append(reverb_noise)
                    #direct_srcs_list.append(direct_noise)
                elif self.noise_type == 'none':
                    noise = clean_srcs_list_align[-1]
                    noise = np.repeat(np.expand_dims(noise, 0), RIR.shape[0], axis=0)
                    reverb_srcs_list.append(noise)
                    direct_srcs_list.append(noise)
                else:
                    raise ValueError("Noise type undefined.")

            reverb_srcs_list, gain_list = self.scale_audios(reverb_srcs_list, snr_list)
            direct_srcs_list = [direct_srcs_list[i] * np.power(10, (gain_list[i] / 20.)) for i in range(len(gain_list))]
            clean_srcs_list_align = [clean_srcs_list_align[i] * np.power(10, (gain_list[i] / 20.)) for i in range(len(gain_list))]

            reverb_srcs = np.stack(reverb_srcs_list, 0)
            direct_srcs = np.stack(direct_srcs_list, 0) 
            clean_srcs = np.stack(clean_srcs_list_align, 0)
            mixture = np.sum(reverb_srcs, 0).transpose()
            #print("reverb_srcs", reverb_srcs.shape)
            #print("clean_srcs", clean_srcs.shape)
            #print("mixture", mixture.shape)
            #raise
        else:
            clean_srcs_list_align, gain_list = self.scale_audios(clean_srcs_list_align, snr_list)
            clean_srcs = np.stack(clean_srcs_list_align, 0)
            mixture = np.sum(clean_srcs, 0)

        if self.normalize:
            max_sample = max(np.max(np.abs(mixture)), np.max(np.abs(clean_srcs)), np.max(np.abs(direct_srcs))) + 1e-12
            clean_srcs = clean_srcs / max_sample
            direct_srcs = direct_srcs / max_sample
            mixture = mixture / max_sample

        src_audio = mixture
        if self.target == 'clean':
            tgt_audio = clean_srcs[:num_spk, :]
        elif self.target == 'direct':
            tgt_audio = direct_srcs[:num_spk, self.ref_channel, :]

        if len(src_audio.shape) == 2:
            src_audio = src_audio[:, self.channel]
        elif len(src_audio.shape) == 1:
            assert self.channel == 0
        else:
            raise ValueError("Invalid audio shape")

        assert src_audio.shape[0] == tgt_audio.shape[1]
        num_samples = src_audio.shape[0]

        #save_dir = "/scratch4/lgarci27/hzili1/workspace/Amazon/fairseq/examples/enhancement/listen"
        #if not os.path.exists(save_dir):
        #    os.makedirs(save_dir)
        #if idx <= 50:
        #    sf.write('{}/{:03d}_mix.wav'.format(save_dir, idx), src_audio, 16000)
        #    sf.write('{}/{:03d}_clean.wav'.format(save_dir, idx), clean_srcs[0, :], 16000)
        #    sf.write('{}/{:03d}_direct.wav'.format(save_dir, idx), direct_srcs[0, 0, :], 16000)

        if self.chunk_size > 0:
            if self.chunk_size < num_samples:
                start_sample = np.random.randint(low=0, high=num_samples-self.chunk_size)
                #print("idx", idx, "uttname_list", "_".join(uttname_list), "snr_list", snr_list, "start_sample_list", start_sample_list, "start_sample", start_sample)
                tgt_audio = tgt_audio[:, start_sample:start_sample+self.chunk_size]
                src_audio = src_audio[start_sample:start_sample+self.chunk_size]
            else:
                tgt_audio = np.pad(tgt_audio, [(0, 0), (0, self.chunk_size - num_samples)])
                if len(src_audio.shape) == 2:
                    src_audio = np.pad(src_audio, [(0, self.chunk_size - num_samples), (0, 0)])
                elif len(src_audio.shape) == 1:
                    src_audio = np.pad(src_audio, (0, self.chunk_size - num_samples))

        src_audio = np.transpose(src_audio)
        #print("src_audio", src_audio.shape)
        #print("tgt_audio", tgt_audio.shape)
        #print("uttname_list", uttname_list)
        return src_audio, tgt_audio, '_'.join(uttname_list)

    def collater(self, batch):
        sorted_batch = sorted(batch, key=lambda x: -x[1].shape[0])
        bs = len(sorted_batch)
        src_audios = torch.stack([torch.from_numpy(sorted_batch[i][0]).float() for i in range(bs)], 0)
        tgt_audios = torch.stack([torch.from_numpy(sorted_batch[i][1]).float() for i in range(bs)], 0)
        uttname_list = [sorted_batch[i][2] for i in range(bs)]
        src_stft = torch.stft(
                src_audios.view(-1, src_audios.size(-1)),
                self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=torch.hann_window(self.win_length).to(src_audios.device),
                return_complex=True
            )
        src_stft = (src_stft.view(bs, -1, src_stft.size(1), src_stft.size(2))).permute(0,1,3,2)
        src_stft_mag = torch.abs(src_stft) + 1e-10
        src_stft_phase = src_stft / src_stft_mag
        
        tgt_stft = torch.stft(
                tgt_audios.view(-1, tgt_audios.size(-1)),
                self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=torch.hann_window(self.win_length).to(tgt_audios.device),
                return_complex=True
            )
        tgt_stft = (tgt_stft.view(bs, -1, tgt_stft.size(1), tgt_stft.size(2))).permute(0,1,3,2)
        tgt_stft_mag = torch.abs(tgt_stft) + 1e-10
        tgt_stft_phase = tgt_stft / tgt_stft_mag

        batch = {
            "src_audios": src_audios,
            "tgt_audios": tgt_audios,
            "src_stft_mag": src_stft_mag,
            "tgt_stft_mag": tgt_stft_mag,
            "src_stft_phase": src_stft_phase,
            "tgt_stft_phase": tgt_stft_phase,
            "uttname_list": uttname_list,
        }
        return batch 

    def size(self, index):
        if self.chunk_size > 0:
            return self.chunk_size
        else:
            return 16000 

    def num_tokens(self, index):
        return self.size(index)

if __name__ == '__main__':
    data_dir = '/scratch4/lgarci27/hzili1/workspace/Amazon/fairseq/examples/enhancement/dataset/test_ami'
    dataset = SeparationDataset(
        data_dir,
        rate=16000,
        src_cond='wav',
        tgt_conds='none',
        channel='0',
        n_fft=512,
        hop_length=160,
        win_length=400,
        chunk_size=-1,
        ref_channel=0,
        shuffle=False,
    )
    for i, v in enumerate(dataset):
        print('-' * 80)
        src_audio, tgt_audio, uttname = v
        print("src_audio", src_audio.shape) 
        print("tgt_audio", tgt_audio)
        print("uttname", uttname)
        if i == 5:
            break
