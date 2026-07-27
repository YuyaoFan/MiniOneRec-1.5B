accelerate launch --num_processes 8 amazon_text2emb.py \
    --dataset Industrial_and_Scientific \
    --root ../../data/Amazon/index \
    --plm_checkpoint your_emb_model_path
