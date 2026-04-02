import os
import numpy as np
import torch
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from qwen3_vl import *
from mydataset import *
from peft import get_peft_model, PeftModel
from peft.utils import set_peft_model_state_dict, get_peft_model_state_dict
from rouge_score import rouge_scorer
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import warnings
import requests
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import TrainingArguments
from transformers import Trainer,EarlyStoppingCallback
from peft.utils import set_peft_model_state_dict, get_peft_model_state_dict
from trl import GRPOTrainer
from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker
from copy import deepcopy
from trl.trainer.utils import selective_log_softmax, entropy_from_logits
from trl.trainer.grpo_trainer import *

def remote_llm_score(pred, ref, server_url="http://localhost:5005/score"):
    try:
        response = requests.post(server_url, json={"pred": pred, "ref": ref}, timeout=10)
        return float(response.json()["score"])
    except Exception as e:
        print(f"[Remote LLM Score Error] {e}")
        return 0.0

# 主 reward 函数，传入预测与参考文本，返回 [reward_1, reward_2, ..., reward_n]
def compute_reward(predictions, references, lambda_weight=0.7):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rewards = []

    for pred, ref in zip(predictions, references):
        print('result')
        print('*************pred*************')
        print(pred)
        print('*************ref*************')
        print(ref)
        # 1. ROUGE-L F1
        rouge_score = scorer.score(ref, pred)["rougeL"].fmeasure

        # 2. LLM-based coherence score
        #llm_score = get_llm_score(pred, ref)
        llm_score = remote_llm_score(pred, ref)
        # 3. 加权融合
        reward = lambda_weight * llm_score + (1 - lambda_weight) * rouge_score
        rewards.append(reward)

    return rewards

def load_all(save_dir, config, tokenizer=None, lora_config=None, inject_lora=False):
    # Step 1: 初始化基础模型（未注入 LoRA）
    model = Qwen3_TextImageModel(config=config, tokenizer=tokenizer, lora_config=None)

    # Step 2: 加载 projector 和 embed 权重
    model.image_projector.load_state_dict(torch.load(f"{save_dir}/image_projector.pt"))
    model.special_token_embed.load_state_dict(torch.load(f"{save_dir}/special_token_embed.pt"))

    if inject_lora and lora_config is not None:
        # Step 3a: 注入 LoRA adapter（重新初始化结构）
        model.text_model = get_peft_model(model.text_model, lora_config)

        # Step 3b: 临时加载保存的 PeftModel，用于提取 LoRA adapter 权重
        print("Loading existing LoRA adapter weights...")
        temp_peft_model = PeftModel.from_pretrained(model.text_model.model, save_dir)
        adapter_state_dict = get_peft_model_state_dict(temp_peft_model)

        # Step 3c: 注入权重到当前模型
        set_peft_model_state_dict(model.text_model, adapter_state_dict)

    else:
        # 不注入 LoRA，只是用来推理或eval的情形
        model.text_model = PeftModel.from_pretrained(model.text_model, save_dir)

    return model

# 你的 LoRA 配置
lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.05
            )

# 自定义 collate
def custom_collate_fn(batch):
    return batch
    # prompts = [item["prompt"] for item in batch]
    # targets = [item["target_text"] for item in batch]
    # image_features = [item["image_features"] for item in batch]
    # task_ids = [item["task_id"] for item in batch]
    #
    # input_ids_list, attention_mask_list, labels_list = [], [], []
    #
    # for prompt, target in zip(prompts, targets):
    #     enc_prompt = tokenizer(prompt, add_special_tokens=False,
    #         padding=False,
    #         truncation=True,
    #         return_tensors="pt")
    #     enc_target = tokenizer(target + tokenizer.eos_token, add_special_tokens=False,
    #         padding=False,
    #         truncation=True,
    #         return_tensors="pt")
    #
    #     input_ids = torch.cat([enc_prompt["input_ids"], enc_target["input_ids"]], dim=1).squeeze(0)
    #     attention_mask = torch.cat([enc_prompt["attention_mask"], enc_target["attention_mask"]], dim=1).squeeze(0)
    #     labels = torch.cat([
    #         torch.full_like(enc_prompt["input_ids"], -100),
    #         enc_target["input_ids"]
    #     ], dim=1).squeeze(0)
    #
    #     input_ids_list.append(input_ids)
    #     attention_mask_list.append(attention_mask)
    #     labels_list.append(labels)
    #
    # input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    # attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    # labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    # image_features = torch.from_numpy(np.array(image_features)).squeeze(1)
    # task_ids = torch.tensor(task_ids)
    #
    # return {
    #     "prompt": prompts,  # 注意：这个必须保留 string 类型
    #     "input_ids": input_ids,
    #     "attention_mask": attention_mask,
    #     "labels": labels,
    #     "image_features": image_features,
    #     "task_id": task_ids
    # }

# tokenizer 初始化
tokenizer = AutoTokenizer.from_pretrained("/data2/liuxj/1-Sentiment-mllm/model_train/best_model_cot3")

# 加载数据集
train_dir = "/data2/liuxj/1-Sentiment-mllm/model_train/data/filter_data/thinking/train2.txt"
eval_dir = "/data2/liuxj/1-Sentiment-mllm/model_train/data/filter_data/thinking/train2.txt"

train_dataset = GrpoDataset(train_dir, tokenizer)
eval_dataset = GrpoDataset(eval_dir, tokenizer)

config = TextImageConfig(text_model_path='Qwen/Qwen3-8B')
model = load_all("/data2/liuxj/1-Sentiment-mllm/model_train/best_model_cot3",
                 config=config,
                 tokenizer=tokenizer,
                 lora_config=lora_config,
                 inject_lora=True)  # 表示你要注入新的LoRA微调

model.text_model.gradient_checkpointing_enable()
model.config.use_cache = False  # 关闭缓存，配合 Trainer 使用

class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, data_collator=None,tokenizer=None,**kwargs):
        self.tokenizer = tokenizer
        self.custom_data_collator = data_collator
        super().__init__(*args, **kwargs)

    def _prepare_inputs(self, inputs):
        # === Step 1: 生成预测结果 ===
        generation_outputs = self._generate_and_score_completions(inputs)
        return generation_outputs

    def _get_per_token_logps_and_entropies(
            self, model, input_ids, attention_mask, logits_to_keep,
            batch_size=None, compute_entropy=False,
            image_features=None, task_id=None):
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []

        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start: start + batch_size]
            attention_mask_batch = attention_mask[start: start + batch_size]
            image_features_batch = (
                image_features[start: start + batch_size]
            )
            task_id_batch = (
                task_id[start: start + batch_size] if isinstance(task_id, torch.Tensor) else task_id
            )

            outputs_all = []
            for i in range(input_ids_batch.size(0)):  # 遍历每个样本（原始batch=2）
                input_ids_i = input_ids_batch[i].unsqueeze(0)  # (1, seq_len)
                attention_mask_i = attention_mask_batch[i].unsqueeze(0)  # (1, seq_len)
                image_features_i = image_features_batch#[i].unsqueeze(0) if image_features_batch is not None else None
                task_id_i = task_id_batch[i].unsqueeze(0) if task_id_batch is not None else None

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # 如果你启用了 bf16
                    output_i = model(
                        input_ids=input_ids_i,
                        attention_mask=attention_mask_i,
                        logits_to_keep=logits_to_keep + 1,
                        image_features=image_features_i,
                        task_id=task_id_i
                    )
                outputs_all.append(output_i)

            logits = torch.cat([o.logits for o in outputs_all], dim=0)  # 合并成 (B, seq_len, vocab)
            # 注意 logits_to_keep + 1 是为了保留对最后一个 token 的预测
            # outputs = model(
            #     input_ids=input_ids_batch,
            #     attention_mask=attention_mask_batch,
            #     logits_to_keep=logits_to_keep + 1,
            #     image_features=image_features_batch,
            #     task_id=task_id_batch)

            #logits = outputs.logits  # (B, L, V)
            logits = logits[:, :-1, :]  # 去掉最后一个 logit（预测的是下一个 token）
            logits = logits / self.temperature  # 除以温度

            # 获取要计算 logprobs 的 token ids
            completion_ids = input_ids_batch[:, -logits_to_keep:].to(logits.device)
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

            if compute_entropy:
                entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return {"logps": logps, "entropies": entropies}

    def _compute_loss(self, model, inputs):
        # === 获取输入 ===
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        image_features = inputs["image_features"]
        task_id = inputs["task_id"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # === 计算 logprobs 和 entropy ===
        if self.token_entropy_percentile_threshold > 0.0:
            logps_and_entropies = self._get_per_token_logps_and_entropies(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep,
                compute_entropy=True,
                image_features=image_features,
                task_id=task_id
            )
            per_token_logps = logps_and_entropies["logps"]
            entropies = logps_and_entropies["entropies"]
            entropy_threshold = torch.quantile(entropies.flatten(), self.token_entropy_percentile_threshold)
            entropy_mask = entropies >= entropy_threshold
        else:
            per_token_logps = self._get_per_token_logps_and_entropies(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep,
                image_features=image_features,
                task_id=task_id
            )["logps"]
            entropy_mask = None

        # === KL 散度 ===
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) -
                    (ref_per_token_logps - per_token_logps) - 1
            )

        # === PPO 主体 ===
        advantages = inputs["advantages"]
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        device = next(model.parameters()).device  # 获取模型所在设备

        advantages = inputs["advantages"].to(device)
        coef_1 = coef_1.to(device)
        coef_2 = coef_2.to(device)
        completion_mask = completion_mask.to(device)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # === Logging ===
        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        # low clip
        gathered_low_clip = self.accelerator.gather(low_clip)
        valid_low_clip = gathered_low_clip[~torch.isnan(gathered_low_clip)]
        min_low_clip = valid_low_clip.min() if valid_low_clip.numel() > 0 else torch.tensor(0.0,
                                                                                            device=gathered_low_clip.device)
        self._metrics[mode]["clip_ratio/low_min"].append(min_low_clip.item())

        # high clip
        gathered_high_clip = self.accelerator.gather(high_clip)
        valid_high_clip = gathered_high_clip[~torch.isnan(gathered_high_clip)]
        max_high_clip = valid_high_clip.max() if valid_high_clip.numel() > 0 else torch.tensor(0.0,
                                                                                  device=gathered_high_clip.device)
        self._metrics[mode]["clip_ratio/high_max"].append(max_high_clip.item())

        gathered_low_clip = self.accelerator.gather(low_clip)
        valid_low_clip = gathered_low_clip[~torch.isnan(gathered_low_clip)]
        low_mean = valid_low_clip.mean() if valid_low_clip.numel() > 0 else torch.tensor(0.0,
                                                                                         device=gathered_low_clip.device)
        self._metrics[mode]["clip_ratio/low_mean"].append(low_mean.item())

        gathered_high_clip = self.accelerator.gather(high_clip)
        valid_high_clip = gathered_high_clip[~torch.isnan(gathered_high_clip)]
        high_mean = valid_high_clip.mean() if valid_high_clip.numel() > 0 else torch.tensor(0.0,
                                                                                            device=gathered_high_clip.device)
        self._metrics[mode]["clip_ratio/high_mean"].append(high_mean.item())

        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        valid_clip_ratio = gathered_clip_ratio[~torch.isnan(gathered_clip_ratio)]
        mean_clip_ratio = valid_clip_ratio.mean() if valid_clip_ratio.numel() > 0 else torch.tensor(0.0,
                                                                                                    device=gathered_clip_ratio.device)
        self._metrics[mode]["clip_ratio/region_mean"].append(mean_clip_ratio.item())

        return loss

    def _score_completions(self, prompts, generated_outputs):
        decoded_preds = [
            self.processing_class.decode(t, skip_special_tokens=True)
            if isinstance(t, torch.Tensor) else t
            for t in generated_outputs
        ]
        references = [p["target_text"] for p in prompts]
        rewards = compute_reward(decoded_preds, references)
        return rewards

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # 保存模型（包括 LoRA 和 projector）
        self.model.text_model.save_pretrained(output_dir)
        torch.save(self.model.image_projector.state_dict(), os.path.join(output_dir, "image_projector.pt"))
        torch.save(self.model.special_token_embed.state_dict(), os.path.join(output_dir, "special_token_embed.pt"))
        self.model.config.save_pretrained(output_dir)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=self.custom_data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker if self.args.seed is not None else None,
        )

    # def _generate_and_score_completions(self, inputs):
    #     """
    #     重写原方法，使得 prompt 附带的 image_features 和 task_id 能传进 generate。
    #     """
    #     # 取出 prompt、image、task_id（你在 dataset 中已经准备好了）
    #     prompts = [x["prompt"] for x in inputs]
    #     image_features = [item["image_features"] for item in inputs]
    #     task_ids = [item["task_id"] for item in inputs]
    #     image_features = torch.from_numpy(np.array(image_features)).squeeze(1)
    #     task_ids = torch.tensor(task_ids)
    #
    #     # tokenizer 生成 input_ids 和 attention_mask
    #     encodings = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    #     input_ids = encodings.input_ids.to(self.accelerator.device)
    #     attention_mask = encodings.attention_mask.to(self.accelerator.device)
    #
    #     # 调用模型的 generate
    #     generated_outputs = self.model.generate(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         image_features=image_features,
    #         task_id=task_ids[0],  # 如果所有 sample 都一样就直接用第一个
    #         **self.generate_kwargs
    #     )
    #
    #     # 后续评估/奖励计算继续使用父类逻辑
    #     return self._score_completions(inputs, generated_outputs)
    def _generate_and_score_completions(self, inputs):
        """
        自定义 GRPO 使用的 completion 生成与打分函数，支持 image_features 与 task_id。
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # 1. 提取 prompt、image_features、task_id
        prompts = [x["prompt"] for x in inputs]
        image_features = torch.from_numpy(np.array([x["image_features"] for x in inputs])).squeeze(1).to(device)
        task_ids = torch.tensor([x["task_id"] for x in inputs], device=device)

        # 2. tokenizer 编码输入
        encodings = self.processing_class(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        prompt_completion_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_features=image_features,
            task_id=task_ids[0],  # 假设 task_ids 相同
            generation_config=self.generation_config
        )

        #print(prompt_completion_ids.shape)

        # 4. 拆分 prompt 与 completion
        #prompt_len = input_ids.size(1)#+image_features.shape[1]
        #print(prompt_len)
        #completion_ids = prompt_completion_ids[:, prompt_len:]
        completion_ids = prompt_completion_ids[:, :]
        prompt_ids = input_ids
        logits_to_keep = completion_ids.size(1)

        # 5. 生成 attention mask（prompt + completion）
        #is_eos = completion_ids == self.processing_class.eos_token_id
        #eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        # 5. 生成 attention mask（prompt + completion）
        is_eos = completion_ids == self.processing_class.eos_token_id

        # 确保 is_eos 的维度正确
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)

        # 检查是否存在 eos_token
        if is_eos.any(dim=1).any():
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        else:
            # 如果没有 eos_token 的话，这里可以做一些处理，例如：设置为0
            eos_idx.fill_(self.processing_class.eos_token_id)  # 这种情况可以替换为你想要的默认值

        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(is_eos.size(1), device=device).expand_as(completion_ids)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
        attention_mask = completion_mask
        #attention_mask = torch.cat([attention_mask, completion_mask], dim=1)

        # 6. 计算 per-token logps 和参考模型 logps
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        with torch.no_grad():
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps_and_entropies(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )["logps"]
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps_and_entropies(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )["logps"]
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps_and_entropies(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )["logps"]
            else:
                ref_per_token_logps = None

        # 7. 解码 completion 文本
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        #print(completions_text)
        # target 扩展（如果 reward 是 per-sample 的，需要扩展 reference）
        references = [x["target_text"] for x in inputs]
        references = [ref for ref in references for _ in range(self.num_generations)]
        rewards = compute_reward(completions_text, references)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        #advantages = torch.tensor(rewards, dtype=torch.float32, device=device)

        # 8. 扩展优势值
        #advantages_expanded = advantages.unsqueeze(1).expand(-1, logits_to_keep)

        # 9. 检查 advantages 的大小，避免 reshape 错误
        # if advantages.size(0) > 1:
        #     # 如果 advantages 的大小大于 1，则可以继续 reshape
        #     mean_grouped_rewards = advantages.view(-1, self.num_generations).mean(dim=1)
        #     std_grouped_rewards = advantages.view(-1, self.num_generations).std(dim=1)
        #     is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))
        #
        #     # 归一化 rewards 以计算 advantages
        #     mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        #     std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        #     advantages = advantages - mean_grouped_rewards
        #     if self.scale_rewards:
        #         advantages = advantages / (std_grouped_rewards + 1e-4)
        # else:
        #     # 如果 advantages 只有一个值，则跳过 reshape 操作
        #     mean_grouped_rewards = advantages
        #     std_grouped_rewards = torch.zeros_like(advantages)
        #
        # # 10. 切分出当前进程的数据
        # process_slice = slice(
        #     self.accelerator.process_index * len(prompts),
        #     (self.accelerator.process_index + 1) * len(prompts),
        # )
        # all_process_advantages = advantages.clone()  # 保存聚合的 advantages 以便后续记录
        #advantages = advantages[process_slice]

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        advantages = advantages[process_slice]
        print('***********advantages**********')
        print(advantages)

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        completion_lengths = completion_mask.sum(1)
        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)

        # 先将 rewards 转换为 Tensor
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        # 只计算有效的（非 NaN 和非 inf）值
        valid_rewards = rewards[torch.isfinite(rewards)]
        # 计算有效值的标准差
        std_rewards = valid_rewards.std().item()
        # 保存标准差到日志
        self._metrics[mode][f"rewards/std"].append(std_rewards)

        # 只计算有效的（非 NaN 和非 inf）值
        valid_rewards = rewards[torch.isfinite(rewards)]
        # 计算有效值的标准差
        std_rewards = valid_rewards.std().item()
        # 保存标准差到日志
        self._metrics[mode][f"rewards/std"].append(std_rewards)
        #self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        #self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        #self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())


        # 9. 返回值（完整用于 loss）
        return {
            "prompt_ids": prompt_ids.repeat_interleave(self.num_generations, dim=0),
            "prompt_mask": input_ids.ne(self.processing_class.pad_token_id).repeat_interleave(self.num_generations, dim=0).long(),
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "ref_per_token_logps": ref_per_token_logps,
            "old_per_token_logps": old_per_token_logps,
            "image_features": image_features,#.repeat_interleave(self.num_generations, dim=0),
            "task_id": task_ids.repeat_interleave(self.num_generations, dim=0),
        }


# 训练配置，num_generations 对应 GRPO 生成次数
training_args = GRPOConfig(
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    max_steps=300,
    num_generations=2,
    gradient_accumulation_steps=8,
    bf16=True,
    deepspeed="deepspeed3.json",
    logging_steps=50,
    save_steps=300,
    output_dir="./checkpoints",
    eval_strategy="no",
    eval_steps=300,
    save_total_limit=3,
)

trainer = CustomGRPOTrainer(
    model=model,
    #ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    reward_funcs=[compute_reward],
    processing_class=tokenizer,
    data_collator=custom_collate_fn,  # ✅ 现在被支持了！
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# trainer.generate_kwargs = {
#             "max_new_tokens": 3000,
#             "do_sample": False,
#             "eos_token_id": tokenizer.eos_token_id,
#             "pad_token_id": tokenizer.pad_token_id,
#             "temperature": 0.7
#         }

# 开始训练
trainer.train()
trainer.save_model("best_model_grpo")
tokenizer.save_pretrained("best_model_grpo")
