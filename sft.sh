export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export OMP_NUM_THREADS=8        # 消掉 libgomp 警告
export CUDA_VISIBLE_DEVICES=0   # 单卡
export WANDB_DISABLED=true      # 关 wandb(无 API key,会卡登录)
# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    torchrun --nproc_per_node 1 \
            sft.py \
            --base_model ./models/Qwen2.5-1.5B \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ./output/sft/${category}_plus \
            --wandb_project wandb_proj \
            --wandb_run_name wandb_name \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/${category}.index.json \
            --item_meta_path ./data/Amazon/index/${category}.item.json \
            --freeze_LLM False
done
