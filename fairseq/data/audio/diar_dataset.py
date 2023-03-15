# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
from typing import Any, List, Optional, Union

import numpy as np
import json

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
import random
import soundfile as sf

random.seed(7)

logger = logging.getLogger(__name__)

def load_mapping(fname):
    utt2info = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        if len(line_split) == 2:
            utt2info[line_split[0]] = line_split[1]
        else:
            utt2info[line_split[0]] = [line_split[i] for i in range(1, len(line_split))]
    return utt2info

def load_utt2spks(fname):
    utt2info = {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt2info[line_split[0]] = [line_split[i] for i in range(1, len(line_split))]
    return utt2info

def load_spk_embed(embed_dir):
    spk2emb = {}
    with open("{}/embed.json".format(embed_dir), 'r') as json_f:
        json_data = json.load(json_f)
    for spk in json_data:
        embed = np.load(json_data[spk])
        spk2emb[spk] = embed
    return spk2emb

class DiarDataset(FairseqDataset):
    def __init__(
        self,
        data_dir: str,
        sample_rate: float,
        embed_dir: str,
        label_rate: int = 50,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        random_crop: bool = False,
        active_p: float = 0.9,
        pad_start_end: bool = True,
        pad_idx: int = -100,
        single_spk: bool = False,
        max_nspks: int = -1,
    ):
        fnames_list = ['wav.scp', 'utt2rttm', 'utt2spks', 'utt2num_samples']
        for f in fnames_list:
            assert os.path.exists("{}/{}".format(data_dir, f))
        self.utt2wav = load_mapping("{}/wav.scp".format(data_dir))
        self.utt2rttm = load_mapping("{}/utt2rttm".format(data_dir))
        self.utt2spks = load_utt2spks("{}/utt2spks".format(data_dir))
        self.utt2num_samples = load_mapping("{}/utt2num_samples".format(data_dir))
        self.audio_names = list(self.utt2wav.keys())
        self.audio_names.sort()
        self.active_p = active_p
        self.sample_rate = sample_rate

        self.sizes = [int(self.utt2num_samples[utt]) for utt in self.audio_names]
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.label_rate = label_rate

        if embed_dir is None:
            self.spk2emb = None
        else:
            self.spk2emb = load_spk_embed(embed_dir) 
            logger.info("Loaded embeddings for {} speakers, embedding dimension {}".format(len(self.spk2emb), list(self.spk2emb.values())[0].shape[1]))

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        self.pad_start_end = pad_start_end
        self.pad_idx = pad_idx
        self.single_spk = single_spk
        self.max_nspks = max_nspks
        if self.max_nspks > 0:
            assert not self.single_spk

    def get_audio(self, index):
        wav_path = self.utt2wav[self.audio_names[index]]
        wav, cur_sample_rate = sf.read(wav_path)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def get_label(self, index):
        rttm_file = self.utt2rttm[self.audio_names[index]]
        seg_list, spk_list = [], []
        with open(rttm_file, 'r') as fh:
            content = fh.readlines()
        for line in content:
            line = line.strip('\n')
            line_split = line.split()
            start_t, end_t = float(line_split[3]), float(line_split[3]) + float(line_split[4])
            start_t, end_t = round(start_t, 2), round(end_t, 2)
            spk = line_split[7]
            seg_list.append([start_t, end_t])
            spk_list.append(spk)
        return seg_list, spk_list

    def convert_label(self, seg_list, spk_list, spks, n_frames):
        label = torch.zeros(len(spks), n_frames)
        for i in range(len(seg_list)):
            if spk_list[i] not in spks:
                continue
            spk_idx = spks.index(spk_list[i])
            start_frame, end_frame = int(np.round(seg_list[i][0] * self.label_rate)), int(np.round(seg_list[i][1] * self.label_rate))
            label[spk_idx, start_frame:end_frame] = 1
        return label

    def get_embed(self, spk):
        if self.spk2emb is None:
            return None
        else:
            assert spk in self.spk2emb, "Speaker name {} not in dictionary".format(spk)
            embed = self.spk2emb[spk]
            embed = torch.from_numpy(embed).float()
            return embed

    def __getitem__(self, index):
        utt = self.audio_names[index]
        wav = self.get_audio(index)
        seg_list, spk_list = self.get_label(index)
        spks_raw = self.utt2spks[utt]
        spks = [spk for spk in spks_raw if spk in self.spk2emb]
        #if len(spks) != len(spks_raw):
        #    s = [spk for spk in spks_raw if spk not in spks]
        #    #print("Warning: Speaker {} not in spk2emb.".format(' '.join(s)))
        embed_list = [self.get_embed(spk) for spk in spks]
        embed = torch.cat(embed_list, dim=0)
        n_frames = int(self.sizes[index] / self.sample_rate * self.label_rate)

        label = self.convert_label(seg_list, spk_list, spks, n_frames)

        if self.single_spk:
            total_frames = torch.sum(label, 1).numpy()
            active, inactive = list(np.nonzero(total_frames != 0)[0]), list(np.nonzero(total_frames == 0)[0])
            assert len(active) + len(inactive) == len(spks)
            if len(active) == 0: # no speaker is active
                spk = random.sample(inactive, 1)[0]
            elif len(inactive) == 0: # every speaker is active
                spk = random.sample(active, 1)[0]
            else:
                if random.random() <= self.active_p:
                    spk = random.sample(active, 1)[0]
                else:
                    spk = random.sample(inactive, 1)[0]
            label = label[spk:spk+1, :]
            embed = embed[spk:spk+1, :]
            wav = torch.unsqueeze(wav, 0)
            #print("active", active, "inactive", inactive)
        else:
            if self.max_nspks > 0:
                assert len(label) == len(embed)
                if len(label) > self.max_nspks:
                    label = label[:self.max_nspks, :]
                    embed = embed[:self.max_nspks, :]
                elif len(label) == self.max_nspks:
                    pass
                else:
                    label_new = torch.zeros(self.max_nspks, label.size(1))
                    embed_new = torch.zeros(self.max_nspks, embed.size(1))
                    label_new[:label.size(0), :] = torch.clone(label)
                    embed_new[:label.size(0), :] = torch.clone(embed)
                    label = label_new
                    embed = embed_new
        
        return {"id": index, "source": wav, "label": label, "embed": embed, "uttname": utt}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audios = self.expand_2Darray(audios)
        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )
        #print("collated_audios", collated_audios.size())

        targets = [s["label"] for s in samples]
        targets = self.expand_2Darray(targets)
        targets, lengths, ntokens = self.collater_label(targets)
        #print("targets", targets.size())
        #print("lengths", lengths)
        #print("ntokens", ntokens)
        
        embeds = [s["embed"] for s in samples]
        embeds = self.expand_2Darray(embeds)
        collated_embeds = self.collater_embed(embeds)
        #print("collated_embeds", collated_embeds.size())

        if self.pad_start_end:
            # this setup is specific to HuBERT/wav2vec2 setup
            collated_audios_new = torch.zeros(collated_audios.size(0), collated_audios.size(1) + 80)
            collated_audios_new[:, 40:-40] = collated_audios
            padding_mask_new = torch.zeros(padding_mask.size(0), padding_mask.size(1) + 80, dtype=torch.bool)
            padding_mask_new[:, 40:-40] = padding_mask
        else:
            collated_audios_new = collated_audios
            padding_mask_new = padding_mask

        net_input = {"source": collated_audios_new, "padding_mask": padding_mask_new, "embed": collated_embeds}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "uttname": [s["uttname"] for s in samples],
            "net_input": net_input,
        }

        batch["target_lengths"] = lengths
        batch["ntokens"] = ntokens
        batch["target"] = targets
        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts

    def collater_label(self, targets):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=self.pad_idx, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_embed(self, embeds):
        if embeds[0] is None:
            collated_embeds = None
        else:
            collated_embeds = torch.stack(embeds, 0)
        return collated_embeds 

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

    def expand_2Darray(self, mat_list):
        new_list = []
        for mat in mat_list:
            for i in range(len(mat)):
                new_list.append(mat[i, :])
        return new_list


if __name__ == '__main__':
    def check_label(label):
        seg_list = []
        state = -1
        start_t = -1
        for i in range(len(label)):
            if state == -1:
                if label[i] == 0:
                    state = 0
                elif label[i] == 1:
                    start_t = i
                    state = 1
            elif state == 0:
                if label[i] == 0:
                    continue
                elif label[i] == 1:
                    start_t = i
                    state = 1
            elif state == 1:
                if label[i] == 0:
                    seg_list.append([start_t, i])
                    state = 0
                elif label[i] == 1:
                    continue
        if state == 1:
            seg_list.append([start_t, len(label)])
        for i in range(len(seg_list)):
            seg_list[i][0] /= 50.0
            seg_list[i][1] /= 50.0
        return seg_list

    data_dir = "minscale/dataset/DIHARD3/dev"
    sample_rate = 16000
    embed_dir = "/export/c05/hzili1/SSL_multispk/embeddings/DIHARD3/xvec/dev"
    active_p = 1.0
    single_spk = True
    bs = 3
    max_nspks = -1
    dataset = DiarDataset(
            data_dir = data_dir,
            sample_rate = sample_rate,
            embed_dir = embed_dir,
            active_p = active_p,
            single_spk = single_spk,
            max_nspks = max_nspks,
        )
    for i, v in enumerate(dataset):
        print('-' * 80)
        # {"id": index, "source": wav, "label": label, "embed": embed}
        print(v['id'])
        print(v['uttname'])
        print('source', v['source'].size())
        print('label', v['label'].size())
        print('embed', v['embed'].size())
        if i == 10:
            break

    #dataloader = torch.utils.data.DataLoader(dataset, \
    #        batch_size=1, \
    #        shuffle=True, \
    #        collate_fn=dataset.collater
    #    )
    #for i, v in enumerate(dataloader):
    #    print('-' * 80)
    #    print(v['uttname'])
    #    print(v['net_input']['source'].size())
    #    print(v['net_input']['padding_mask'].size())
    #    print(v['net_input']['embed'].size())
    #    print(v['target'].size())
    #    for j in range(len(v['target'])):
    #        print(check_label(v['target'][j, :]))
    #    if i == 1:
    #        break
