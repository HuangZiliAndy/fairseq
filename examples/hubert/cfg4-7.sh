#!/bin/bash
#SBATCH -A lgarci27_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20480
#SBATCH --partition=a100
#SBATCH --job-name=wavlm
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -G 1
#SBATCH --exclude=gpu12

export PATH="/scratch4/lgarci27/hzili1/anaconda3/envs/huggingface/bin:$PATH"

hubert_path=/scratch4/lgarci27/hzili1/workspace/Amazon/fairseq/examples/hubert

for cfgname in cfg4 cfg5 cfg6 cfg7; do
  exp_dir=${hubert_path}/exp/${cfgname}
  cfg_path=${hubert_path}/myconfig/finetune
  
  python ../../fairseq_cli/hydra_train.py \
    --config-dir ${cfg_path} \
    --config-name ${cfgname} \
    task.data=${hubert_path}/dataset/AMI/data \
    task.label_dir=${hubert_path}/dataset/AMI/label \
    hydra.run.dir=${exp_dir}
done
