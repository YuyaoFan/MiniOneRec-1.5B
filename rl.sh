#!/bin/bash

export NCCL_IB_DISABLE=1                              # 完全禁用 IB/RoCE
export OMP_NUM_THREADS=8                              # 消掉 libgomp 警告
export CUDA_VISIBLE_DEVICES=0                         # 单卡
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 抗显存碎片

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file}

    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --config_file ./config/zero2_opt.yaml \
                                    --num_processes 1 --main_process_port 29503 \
                                    rl.py \
                        --model_path ./output/sft/${category}_plus \
                        --train_batch_size 32 \
                        --eval_batch_size 32 \
                        --num_train_epochs 2 \
                        --gradient_accumulation_steps 32 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train False \
                        --eval_step 0.0999 \
                        --reward_type ranking \
                        --num_generations 16 \
                        --mask_all_zero False \
                        --dynamic_sampling False \
                        --sync_ref_model True \
                        --beam_search True \
                        --test_during_training False \
                        --temperature 1.0 \
                        --learning_rate 1e-5 \
                        --add_gt False \
                        --beta 1e-3 \
                        --dapo False \
                        --output_dir ./output/rl/${category}_plus \
                        --wandb_run_name rl_${category}_plus \
                        --sid_index_path ./data/Amazon/index/${category}.index.json \
                        --item_meta_path ./data/Amazon/index/${category}.item.json
done
