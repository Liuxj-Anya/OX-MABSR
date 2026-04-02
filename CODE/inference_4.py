import torch
import numpy as np
from transformers import AutoTokenizer
from qwen3_vl import Qwen3_TextImageModel, TextImageConfig
from peft import LoraConfig
from peft import PeftModel
import os

def load_all(save_dir, config, tokenizer=None, lora_config=None):
    # 初始化主模型（包含 projector 和 embed）
    model = Qwen3_TextImageModel(config=config, tokenizer=tokenizer, lora_config=lora_config)
    # 加载 projector 和 special_token_embed 权重
    model.image_projector.load_state_dict(torch.load(f"{save_dir}/image_projector.pt",map_location='cuda:0'))
    model.special_token_embed.load_state_dict(torch.load(f"{save_dir}/special_token_embed.pt",map_location='cuda:0'))
    # 加载 LoRA adapter
    model.text_model = PeftModel.from_pretrained(model.text_model, save_dir)

    return model

# 1. 设置 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/data2/liuxj/1-Sentiment-mllm/model_train/best_model6")

# 3. 加载 config 和 LoRA config
config = TextImageConfig(text_model_path="/data2/liuxj/1-Sentiment-mllm/Qwen/Qwen3-14B")

# 4. 加载模型
model = load_all("/data2/liuxj/1-Sentiment-mllm/model_train/best_model6", config=config, tokenizer=tokenizer, lora_config=None)
model = model.to(device)
model.eval()



def infer_all(text,image_array):
    input_text = "Given an image and a sentence: [" + text + "] , Please think step by step, obtain the sentiments and causes of the entities from multiple levels and angles, and finally output the list of entities and sentiments."
    task_id = 6
    image_features = torch.from_numpy(image_array)
    input_enc = tokenizer(
                f"<s><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{input_text}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
                add_special_tokens=False,
                padding=False,
                truncation=True,
                return_tensors="pt"
                )

    input_ids = input_enc['input_ids']
    attention_mask = input_enc['attention_mask']
    # 7. 生成输出
    with torch.no_grad():
        outputs = model.generate_with_image(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_features=image_features,
            generation_kwargs={
                "max_new_tokens": 3000,
                "do_sample": False,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "temperature": 0.7
            },
            task_id = task_id
        )

    # 8. 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print("模型输出：", response)
    return response

with open('result/task4.txt','a') as fa:
    with open('/data2/liuxj/1-Sentiment-mllm/model_train/data/filter_data/thinking/test.txt') as fp:
        line = fp.readline()
        while line:
            id,_ = line.strip().split('\t')
            with open('data/text/' + id + '.txt', 'r', encoding='utf-8') as ft:
                text = ft.readline().strip()
            image_array = np.load('data/image_feature/' + id + '.npy')
            result = infer_all(text,image_array)
            fa.write(id)
            fa.write('\t')
            fa.write(result)
            fa.write('\n')
            line = fp.readline()
