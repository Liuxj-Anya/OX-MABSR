import difflib
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import numpy as np
from transformers import AutoTokenizer
from qwen3_vl import Qwen3_TextImageModel, TextImageConfig
from peft import LoraConfig
from peft import PeftModel
import ast
import re
from tqdm import tqdm
import difflib
from sklearn.metrics import precision_score, recall_score, f1_score

def fuzzy_match(a, b, threshold=0.8):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def evaluate_aspect_sentiment_fuzzy(predictions, ground_truth, threshold=0.8):
    """
    :param predictions: list of (aspect, sentiment)
    :param ground_truth: list of (aspect, sentiment)
    :param threshold: fuzzy string matching threshold
    :return: evaluation metrics dict
    """

    # ===== 严格匹配指标 =====
    pred_set = set(predictions)
    gt_set = set(ground_truth)

    tp_strict = len(pred_set & gt_set)
    fp_strict = len(pred_set - gt_set)
    fn_strict = len(gt_set - pred_set)

    precision_strict = tp_strict / (tp_strict + fp_strict) if tp_strict + fp_strict else 0
    recall_strict = tp_strict / (tp_strict + fn_strict) if tp_strict + fn_strict else 0
    f1_strict = 2 * precision_strict * recall_strict / (precision_strict + recall_strict + 1e-8) if (precision_strict + recall_strict) else 0

    # ===== 模糊匹配评估 =====
    matched_pred_indices = set()
    matched_gt_indices = set()
    y_true, y_pred = [], []

    fuzzy_tp = 0
    fuzzy_fp = 0
    fuzzy_fn = 0

    for i, (gt_aspect, gt_sentiment) in enumerate(ground_truth):
        matched = False
        for j, (pred_aspect, pred_sentiment) in enumerate(predictions):
            if j in matched_pred_indices:
                continue
            if fuzzy_match(gt_aspect, pred_aspect, threshold=threshold) and gt_sentiment == pred_sentiment:
                matched_pred_indices.add(j)
                matched_gt_indices.add(i)
                fuzzy_tp += 1
                y_true.append(gt_sentiment)
                y_pred.append(pred_sentiment)
                matched = True
                break
        if not matched:
            fuzzy_fn += 1

    fuzzy_fp = len(predictions) - len(matched_pred_indices)

    fuzzy_match_precision = fuzzy_tp / (fuzzy_tp + fuzzy_fp) if fuzzy_tp + fuzzy_fp else 0
    fuzzy_match_recall = fuzzy_tp / (fuzzy_tp + fuzzy_fn) if fuzzy_tp + fuzzy_fn else 0
    fuzzy_match_f1 = 2 * fuzzy_match_precision * fuzzy_match_recall / (fuzzy_match_precision + fuzzy_match_recall + 1e-8) if fuzzy_match_precision + fuzzy_match_recall else 0

    return {
        "fuzzy_match_precision": round(fuzzy_match_precision, 4),
        "fuzzy_match_recall": round(fuzzy_match_recall, 4),
        "fuzzy_match_f1": round(fuzzy_match_f1, 4),
    }
# predictions=[('aa','pos'),('bb','pos')]
# ground_truth=[('bb','pos')]
# result=evaluate_aspect_sentiment_fuzzy(predictions, ground_truth, threshold=0.5)
# print(result)
def load_all(save_dir, config, tokenizer=None, lora_config=None):
    # 初始化主模型（包含 projector 和 embed）
    model = Qwen3_TextImageModel(config=config, tokenizer=tokenizer, lora_config=lora_config)
    # 加载 projector 和 special_token_embed 权重
    model.image_projector.load_state_dict(torch.load(f"{save_dir}/image_projector.pt"))
    model.special_token_embed.load_state_dict(torch.load(f"{save_dir}/special_token_embed.pt"))
    # 加载 LoRA adapter
    model.text_model = PeftModel.from_pretrained(model.text_model, save_dir)

    return model


# 1. 设置 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/data2/liuxj/1-Sentiment-mllm/model_train/best_model_sft")

# 3. 加载 config 和 LoRA config
config = TextImageConfig(text_model_path="Qwen/Qwen3-8B")

# 4. 加载模型
model = load_all("/data2/liuxj/1-Sentiment-mllm/model_train/best_model_sft", config=config, tokenizer=tokenizer, lora_config=None)
model = model.to(device)
model.eval()

with open('/data2/liuxj/1-Sentiment-mllm/model_train/data/twitter/test_17.txt','r',encoding='utf-8') as ft:
    line = ft.readline()
    all_p=0
    all_r=0
    all_f1=0
    count = 0
    while line:
        id,answer,text = line.strip().split('\t')
        image_array = np.load(f'data/twitter_image/{id}.npy')
        input_text = "Extract Explicit entity and its Simple Sentiment(POS or NEU or NEG): Given an image and a sentence: [" + text + "] , please identify the entities and sentiment in the sentence. Output the results in the following format: (Entity, Simple Sentiment)."
        task_id=2

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
                    "max_new_tokens": 500,
                    "do_sample": False,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                    "temperature": 0.7},
                task_id = task_id)

        # 8. 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #print("模型输出：", response)
        # 给未加引号的 sentiment 加引号
        response = re.sub(r'\(\s*(".*?")\s*,\s*([A-Z]+)\s*\)', r'(\1, "\2")',response)
        pred = ast.literal_eval(response)

        answer = re.sub(r'\(\s*(".*?")\s*,\s*([A-Z]+)\s*\)', r'(\1, "\2")',answer)
        gt = ast.literal_eval(answer)

        result = evaluate_aspect_sentiment_fuzzy(pred, gt, threshold=0.5)

        p=result['fuzzy_match_precision']
        r=result['fuzzy_match_recall']
        f1=result['fuzzy_match_f1']
        count+=1
        all_p = all_p +p
        all_r = all_r +r
        all_f1 = all_f1+f1
        line = ft.readline()

print(all_f1/count)
print(all_p/count)
print(all_r/count)


