from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import random
import numpy as np
import torch
from data import D3Dataset, SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset, RLSid2TitleDataset, RLSidhis2TitleDataset
from torch.utils.data import ConcatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from minionerec_trainer import ReReTrainer
from sasrec import SASRec
from fire import Fire
import pickle
import math
import json
from sklearn.metrics import ndcg_score

os.environ['WANDB_MODE'] = 'disabled'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import shutil
from transformers import TrainerCallback

class SaveEvalSnapshotCallback(TrainerCallback):
    """每次存完 checkpoint，把评测需要的小文件(权重+config+tokenizer)拷到
    output_dir/eval_snapshots/checkpoint-N/，跳过优化器/deepspeed 等大文件。
    这样续训只靠 save_total_limit=1 的完整 checkpoint(含优化器)，
    而每个 checkpoint 的 ~3G 权重都永久保留下来供逐个评测。"""

    # 续训需要、评测不需要的大文件/目录前缀，一律跳过
    _SKIP_PREFIX = ("optimizer", "global_step", "rng_state")
    _SKIP_NAME = ("scheduler.pt", "latest", "zero_to_fp32.py", "trainer_state.json", "training_args.bin")
    # 评测时 from_pretrained 必需的文件，缺了就从 fallback_dir(SFT 源)补
    _ESSENTIAL = ("config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "special_tokens_map.json",
                  "added_tokens.json", "vocab.json", "merges.txt")

    def __init__(self, fallback_dir=None):
        self.fallback_dir = fallback_dir

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            return
        dst = os.path.join(args.output_dir, "eval_snapshots", f"checkpoint-{state.global_step}")
        os.makedirs(dst, exist_ok=True)
        copied = 0
        for name in os.listdir(ckpt_dir):
            src = os.path.join(ckpt_dir, name)
            if os.path.isdir(src):                       # 跳过 global_step* 等目录
                continue
            if name.startswith(self._SKIP_PREFIX) or name in self._SKIP_NAME:
                continue
            shutil.copy2(src, os.path.join(dst, name))   # 含 model.safetensors / *.json / tokenizer
            copied += 1
        # 补齐评测必需但 checkpoint 里可能缺的文件
        if self.fallback_dir and os.path.isdir(self.fallback_dir):
            for name in self._ESSENTIAL:
                tgt = os.path.join(dst, name)
                fb = os.path.join(self.fallback_dir, name)
                if (not os.path.exists(tgt)) and os.path.exists(fb):
                    shutil.copy2(fb, tgt)
                    copied += 1
        print(f"[eval-snapshot] saved {copied} files -> {dst}")

def train(
    # model/data params
    model_path: str = "",
    seed: int = 42,
    train_file: str = "",
    eval_file: str = "",
    info_file: str = "",
    category: str = "",
    
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    
    # training hyperparams
    output_dir: str = "",
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    temperature: float = 1.0,
    add_gt: bool = False,
    eval_step: float = 0.199,
    num_generations: int = 16,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-6,
    beta: float = 0.04,
    beam_search: bool = False,
    test_during_training: bool = True,
    dynamic_sampling: bool = False,
    mask_all_zero: bool = False,
    sync_ref_model: bool = False,
    test_beam: int = 20,
    reward_type: str = "rule",
    sample_train: bool = False,
    ada_path: str = "",
    cf_path: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    dapo: bool = False,
    gspo: bool = False,
    gdpo: bool = False,
):
    torch.backends.cuda.enable_flash_sdp(False)  
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    set_seed(seed)
    
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)
    
    
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Extract semantic_id (first column) from the format: semantic_id \t item_title \t item_id
        item_name = [_.split('\t')[0].strip() for _ in info]
        item2id = {name: i for i, name in enumerate(item_name)}

    sample = -1
    train_datasets = []
    # train_data = D3Dataset(train_file, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data)
    train_data1 = SidDataset(train_file, category=category_dict[category], sample=sample)
    train_datasets.append(train_data1)
    train_data2 = RLTitle2SidDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    train_datasets.append(train_data2)
    train_data3 = RLSeqTitle2SidDataset(train_file, category=category_dict[category], sample=10000)
    train_datasets.append(train_data3)
    # train_data4 = RLSid2TitleDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data4)
    # train_data5 = RLSidhis2TitleDataset(train_file, item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data5)
    # train_data6 = RLTitle2Sid_1LayerDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data6)
    # train_data7 = RLTitle2Sid_2LayerDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data7)
    train_data = ConcatDataset(train_datasets)
    # eval_data = D3Dataset(eval_file, category=category_dict[category], sample=sample)
    eval_data = SidDataset(eval_file, category=category_dict[category], sample=sample)

    train_dataset = Dataset.from_dict({k : [elm[k] for elm in train_data] for k in train_data[0].keys()})
    train_dataset = train_dataset.shuffle(seed=seed) 
    if sample_train and "sft" in model_path:
        train_dataset = train_dataset.select(range(int(0.2 * len(train_dataset)), len(train_dataset)))
    eval_dataset = Dataset.from_dict({k : [elm[k] for elm in eval_data] for k in eval_data[0].keys()})
    eval_dataset = eval_dataset.shuffle(seed=seed)
    

    # prompt2history = {**train_data.prompt2history, **eval_data.prompt2history}
    # history2target = {**train_data.history2target, **eval_data.history2target}

    prompt2history = {}
    history2target = {}
    
    # Collect prompt2history and history2target from all train datasets
    for dataset in train_datasets:
        if hasattr(dataset, 'prompt2history'):
            prompt2history.update(dataset.prompt2history)
        if hasattr(dataset, 'history2target'):
            history2target.update(dataset.history2target)
    
    # Add eval_data mappings
    if hasattr(eval_data, 'prompt2history'):
        prompt2history.update(eval_data.prompt2history)
    if hasattr(eval_data, 'history2target'):
        history2target.update(eval_data.history2target)

    print("train_dataset: ", train_dataset)
    print("eval_dataset: ", eval_dataset)

    llm_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    device = llm_model.device
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    len_seq = 10
    item_num = len(item_name)
    print(f"item_num: {item_num}")

    if reward_type == "sasrec":
        model = SASRec(32, item_num, len_seq, 0.3, device)
        model.to(device)
        model.load_state_dict(torch.load(cf_path))
        model.eval()
    if reward_type == "semantic":
        with open(ada_path, "rb") as f:
            item_ada_embd = pickle.load(f)
        item_ada_embd = torch.tensor(item_ada_embd).to(llm_model.device)

    print("Load item_ada_embd successfully.")

    ndcg_rewards = [-1.0/math.log2(i+2) for i in range(num_generations)]
    ndcg_rewards = [-elm/sum(ndcg_rewards) for elm in ndcg_rewards]


    def ndcg_rule_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        repeat = num_generations
        rewards = []
        flag = False
        lis = []

        for i, completion in enumerate(completions):

            if completion.strip("\n\"") == targets[i].strip("\n\""):
                flag = True
                lis.append(0.0)
            else:
                lis.append(ndcg_rewards[i%num_generations])
            
            if (i+1)%num_generations == 0:
                if flag:
                    rewards.extend(lis)
                else:
                    rewards.extend([0.0] * repeat)
                flag = False
                lis = []
        
        return rewards

    def rule_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        rewards = []

        for i, completion in enumerate(completions):

            if completion.strip("\n\" ") == targets[i].strip("\n\" "):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    def semantic_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        target_ids = [item2id[elm.strip("\"\n")] for elm in targets]
        completions = [elm.strip("\"\n") for elm in completions]
        for i, completion in enumerate(completions):
            if completion not in item2id:
                print("==============================")
                print(prompts[i])
                print(f"Invalid item: {completion}")
                print("==============================")
        completion_ids = [item2id[elm] for elm in completions]
        rewards =  torch.cosine_similarity(item_ada_embd[target_ids], item_ada_embd[completion_ids], dim=-1)
        print(rewards)
        return rewards

    def cf_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        history_list = [elm.split("::") for elm in history]
        pred_ids = []
        for i, elm in enumerate(completions):
            elm = elm.strip("\n\"")
            if elm not in item_name:
                # print("========Invalid Item========")
                # print(f"Invalid item: {elm}")
                # print(f"Prompt: {prompts[i]}")
                # print("============================")
                pred_ids.append(random.randint(0, item_num-1))
            else:
                pred_ids.append(item2id[elm])
        
        len_lis = []
        history_ids = []
        for his in history_list:
            his = [item2id[elm] for elm in his]
            len_lis.append(len(his))
            if len(his) < len_seq: 
                his = his + [item_num] * (len_seq - len(his))
            history_ids.append(his)
        
        seq = torch.LongTensor(history_ids).to(device)
        pred = torch.LongTensor(pred_ids).to(device)    
        
        with torch.no_grad():
            predictions = model.forward_eval(seq, torch.tensor(np.array(len_lis)).to(device))
            scores = torch.gather(predictions, 1,  pred.view(-1, 1)).view(-1)
        return scores
    


    if reward_type == "rule":
        reward_fun = rule_reward
    elif reward_type == "ranking":
        reward_fun = [rule_reward, ndcg_rule_reward]
    elif reward_type == "ranking_only":
        reward_fun = ndcg_rule_reward
    elif reward_type == "semantic":
        reward_fun = semantic_reward
    elif reward_type == "sasrec":
        reward_fun = cf_reward
    
    os.environ['WANDB_PROJECT'] = wandb_project
    os.environ["WANDB_MODE"] = "offline"

    training_args = GRPOConfig(output_dir=output_dir,
                                save_steps=0.1,
                                save_total_limit=1,     # 只滚动保留最新 1 个完整 checkpoint(够续训；HF 先写新再删旧，崩在写盘途中旧的仍在)
                                eval_strategy="steps",
                                max_completion_length=128,
                                num_generations=num_generations,
                                temperature=temperature,
                                sync_ref_model=sync_ref_model,
                                per_device_eval_batch_size=eval_batch_size,
                                per_device_train_batch_size=train_batch_size,
                                gradient_accumulation_steps=gradient_accumulation_steps,  
                                eval_steps=eval_step, 
                                logging_steps=1, 
                                learning_rate=learning_rate,
                                beta=beta,
                                warmup_ratio=0.03,
                                max_grad_norm= 0.3,
                                num_train_epochs=num_train_epochs,
                                bf16=True,
                                optim="paged_adamw_32bit",
                                lr_scheduler_type="cosine", 
                                save_strategy="steps",
                                report_to="tensorboard",
                                run_name=wandb_run_name,
                            )
    trainer = ReReTrainer(
        model=model_path,
        base_model=model_path,
        dapo=dapo,
        gspo=gspo,
        gdpo=gdpo,
        add_gt=add_gt,
        dynamic_sampling=dynamic_sampling,
        beam_search=beam_search,
        test_during_training=test_during_training,
        test_beam=test_beam,
        info_file=info_file,
        prompt2history=prompt2history,
        history2target=history2target,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    trainer.add_callback(SaveEvalSnapshotCallback(fallback_dir=model_path))

    # 自动续训：若 output_dir 下已有 checkpoint-*，则从最新一个恢复（含优化器/调度器/step），
    # 否则全新开始。这样崩了重跑同一条命令即可断点续训。
    import glob as _glob
    _ckpts = _glob.glob(os.path.join(output_dir, "checkpoint-*"))
    _resume = True if _ckpts else None
    if _resume:
        print(f"[resume] found {len(_ckpts)} checkpoint(s) in {output_dir}, resuming from latest")
    else:
        print(f"[resume] no checkpoint in {output_dir}, starting fresh")
    trainer.train(resume_from_checkpoint=_resume)

    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
if __name__ == "__main__":
    Fire(train)
