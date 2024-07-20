#!/bin/bash

#SBATCH --job-name=peft
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1 
#SBATCH --partition=gpu-a100 
#SBATCH --time=3-00:00:00
#SBATCH --mem=20480
#SBATCH --exclude=e04
#SBATCH --account=a100acct

module load cuda/12.1
export PYTHONPATH="/export/c12/hzili1/workspace/Amazon/fairseq:$PYTHONPATH"
export PATH="/home/hzili1/anaconda3/envs/peft/bin:$PATH"

hubert_path=/export/c12/hzili1/workspace/Amazon/fairseq/examples/hubert

for cfgname in cfg18 cfg19 cfg20 cfg21; do
  exp_dir=${hubert_path}/exp/${cfgname}
  cfg_path=${hubert_path}/myconfig/finetune
  
  python ../../fairseq_cli/hydra_train.py \
    --config-dir ${cfg_path} \
    --config-name ${cfgname} \
    task.data=${hubert_path}/dataset/AMI/data \
    task.label_dir=${hubert_path}/dataset/AMI/label \
    hydra.run.dir=${exp_dir}
done
