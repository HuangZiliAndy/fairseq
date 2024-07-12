#!/bin/bash
#SBATCH -A lgarci27_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20480
#SBATCH --partition=a100
#SBATCH --job-name=wavlm
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -G 1
#SBATCH --exclude=gpu12

export PATH="/scratch4/lgarci27/hzili1/anaconda3/envs/huggingface/bin:$PATH"
export PYTHONPATH="/scratch4/lgarci27/hzili1/workspace/Amazon/fairseq:${PYTHONPATH}"

hubert_path=/scratch4/lgarci27/hzili1/workspace/Amazon/fairseq/examples/hubert

cfgname=cfg3
for ckpt in checkpoint_best checkpoint_last; do
  ckpt_path=${hubert_path}/exp/${cfgname}/checkpoints/${ckpt}.pt

  python ../../examples/speech_recognition/new/infer.py \
    --config-dir ${hubert_path}/config/decode \
    --config-name infer_viterbi \
    task.data=${hubert_path}/dataset/AMI/data \
    task.label_dir=${hubert_path}/dataset/AMI/label \
    task.normalize=false \
    common_eval.path=${ckpt_path} \
    common_eval.results_path=${hubert_path}/exp/${cfgname}/decode/${ckpt} \
    dataset.gen_subset=test
done
