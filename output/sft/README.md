# SFT 模型存放目录

请从 HuggingFace 下载 sft_model/ 全部文件放入 final_checkpoint/:
  https://huggingface.co/onesfour/MiniOneRec-1.5B-SFT-GDPO/tree/main/sft_model

目录结构:
  output/sft/Industrial_and_Scientific_baseline/final_checkpoint/
    ├── model.safetensors       (2.9 GB)
    ├── config.json
    ├── tokenizer.json / tokenizer_config.json
    ├── vocab.json / merges.txt
    ├── special_tokens_map.json / added_tokens.json
    ├── generation_config.json
    └── chat_template.jinja
