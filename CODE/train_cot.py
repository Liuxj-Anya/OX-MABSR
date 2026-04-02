import os
from qwen3_vl import *
from mydataset import *
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import TrainingArguments
from transformers import Trainer,EarlyStoppingCallback
from peft.utils import set_peft_model_state_dict, get_peft_model_state_dict
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["embed_tokens","q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.05
            )

def load_all(save_dir, config, tokenizer=None, lora_config=None, inject_lora=False):
    # Step 1: 初始化基础模型（未注入 LoRA）
    model = Qwen3_TextImageModel(config=config, tokenizer=tokenizer, lora_config=None)

    # Step 2: 加载 projector 和 embed 权重
    model.image_projector.load_state_dict(torch.load(f"{save_dir}/image_projector.pt",map_location='cuda:0'))
    model.special_token_embed.load_state_dict(torch.load(f"{save_dir}/special_token_embed.pt",map_location='cuda:0'))

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

def custom_collate_fn(batch):

    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    image_features = [item["image_features"] for item in batch]
    task_id = [item["task_id"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    image_features = torch.from_numpy(np.array(image_features)).squeeze(1)
    task_id = torch.tensor(task_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "image_features": image_features,
        "task_id": task_id
    }

tokenizer = AutoTokenizer.from_pretrained("/data2/liuxj/1-Sentiment-mllm/model_train/best_model_cot3")

# 加载数据集
train_dir = "/data2/liuxj/1-Sentiment-mllm/model_train/data/filter_data/thinking/train2.txt"
eval_dir = "/data2/liuxj/1-Sentiment-mllm/model_train/data/filter_data/thinking/test.txt"

train_dataset = MyDataset(train_dir, tokenizer)
eval_dataset = MyDataset(eval_dir, tokenizer)

config = TextImageConfig(text_model_path='Qwen/Qwen3-8B')
model = load_all("/data2/liuxj/1-Sentiment-mllm/model_train/best_model_cot3",
                 config=config,
                 tokenizer=tokenizer,
                 lora_config=lora_config,
                 inject_lora=True)  # 表示你要注入新的LoRA微调
# 启用 gradient checkpointing，节省显存
model.text_model.gradient_checkpointing_enable()
model.config.use_cache = False  # 关闭缓存，配合 Trainer 使用时建议关闭

def save_all(model, save_dir, tokenizer=None):
    # 保存 LoRA adapter 参数
    model.text_model.save_pretrained(save_dir)

    # 保存 projector 和 special_token_embed
    torch.save(model.image_projector.state_dict(), f"{save_dir}/image_projector.pt")
    torch.save(model.special_token_embed.state_dict(), f"{save_dir}/special_token_embed.pt")

    # 保存 config
    model.config.save_pretrained(save_dir)

    # 保存 tokenizer（如果传入）
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)

class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 保存 tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # 保存模型权重（包括 LoRA 和额外组件）
        save_all(self.model, output_dir, tokenizer=self.tokenizer)


# 训练参数
# training_args = TrainingArguments(
#     output_dir="./checkpoints",
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=2,
#     num_train_epochs=5,
#     eval_strategy="steps",
#     eval_steps=500,
#     save_strategy="steps",
#     save_steps=500,
#     save_total_limit=3,
#     logging_steps=50,
#     learning_rate=1e-5,
#     weight_decay=0.01,
#     fp16=False,            # 若你用 bf16，把这个关掉
#     bf16=True,             # 如果你用 A100 建议启用
#     deepspeed="deepspeed_config.json",
#     report_to="none"
# )
#
# trainer = CustomTrainer(
#     model=model,                                 # 你的 Qwen3_TextImageModel
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,                   # 必须有 eval_dataset 才能保存 best model
#     tokenizer=tokenizer,                         # 用于保存/加载和 decode
#     data_collator=custom_collate_fn,             # 自定义 collate，用于 image_features 等处理
#     # + EarlyStoppingCallback(early_stopping_patience=3) 如果想加 Early Stop
# )

# 开始训练

training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=3,
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    deepspeed="deepspeed_config.json",
    report_to="none",
    #label_smoothing_factor=0.2,

    #EarlyStopping
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)


# 实例化 Trainer，添加 EarlyStoppingCallback
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=custom_collate_fn,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=5)  # ✅ 添加动态评估Callback
    ])
trainer.train()
trainer.save_model("best_model_grpo")          # 将当前 model（已是最佳）保存到 best_model/
tokenizer.save_pretrained("best_model_grpo")   # 保存 tokenizer（包含特定 vocab，如 <img>、</img>）

# 评估并提取 eval_loss
eval_metrics = trainer.evaluate()
eval_loss = eval_metrics.get("eval_loss", None)

# 保存 eval_loss 到 best_model/metrics.json
if eval_loss is not None:
    import json
    with open("best_model_grpo/metrics.json", "w") as f:
        json.dump({"eval_loss": eval_loss}, f, indent=2)