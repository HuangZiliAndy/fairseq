#!/bin/bash
#SBATCH -A lgarci27_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20480
#SBATCH --partition=a100
#SBATCH --job-name=enh
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -G 1
#SBATCH --exclude=gpu12

export PATH="/scratch4/lgarci27/hzili1/anaconda3/envs/huggingface/bin:$PATH"

enh_path=/scratch4/lgarci27/hzili1/workspace/Amazon/fairseq/examples/enhancement

for cfgname in enh_cfg3 enh_cfg4 enh_cfg5; do
  exp_dir=${enh_path}/exp/${cfgname}
  cfg_path=${enh_path}/myconfig
  
  python ../../fairseq_cli/hydra_train.py \
    --config-dir ${cfg_path} \
    --config-name ${cfgname} \
    hydra.run.dir=${exp_dir}
done
