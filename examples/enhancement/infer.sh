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

enh_path=/scratch4/lgarci27/hzili1/workspace/Amazon/fairseq/examples/enhancement

for cfgname in enh_cfg10; do 
  for ckpt in checkpoint_best; do
    ckpt_path=${enh_path}/exp/${cfgname}/checkpoints/${ckpt}.pt
  
    python infer.py \
      --config-dir ${enh_path}/myconfig \
      --config-name infer \
      common_eval.path=${ckpt_path} \
      common_eval.results_path=${enh_path}/exp/${cfgname}/decode/${ckpt} \
      dataset.gen_subset=test_ami \
      task.target="none" \
      save_audio=true
  done
done
