import warnings
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# === 加载模型 ===
tokenizer = AutoTokenizer.from_pretrained(
    "/data2/liuxj/1-Sentiment-mllm/Qwen/Qwen3-14B", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "/data2/liuxj/1-Sentiment-mllm/Qwen/Qwen3-14B",
    device_map={"": 0},  # 改为你实际用的 GPU 编号（如 3）
    torch_dtype="auto",
    trust_remote_code=True
)

# === 评分函数 ===
def get_llm_score(pred, ref):
    prompt = f"""Given the following two cognitive sentiment causes, score how semantically consistent, logically coherent, and sentiment aligned the prediction is compared to the gold.

    Gold: {ref}
    Prediction: {pred}

    Return only a number between 0 and 1 (inclusive). Do not include any explanation or extra text. Just output the number.
    Score:"""

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=50
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    output = tokenizer.decode(output_ids[:], skip_special_tokens=True).strip("\n")

    print(output)
    match = re.search(r"\d*\.?\d+", output)
    if match:
        score = float(match.group())
        return min(max(score, 0.0), 1.0)
    else:
        warnings.warn(f"[get_llm_score] No valid score found in output: {output}")
        print('warning')
        return 0.5

# === 文本读取函数 ===
def load_id2text(file_path):
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            id_, text = line.strip().split('\t', 1)
            data[id_] = text
    return data

# === 计算平均LLM分数 ===
def compute_avg_llm_score(ref_file, pred_file):
    ref_data = load_id2text(ref_file)
    pred_data = load_id2text(pred_file)

    assert ref_data.keys() == pred_data.keys(), "ID 不匹配！"

    all_scores = []
    for id_ in ref_data:
        ref = ref_data[id_]
        pred = pred_data[id_]
        score = get_llm_score(pred, ref)
        all_scores.append(score)

    avg_score = sum(all_scores) / len(all_scores)
    return avg_score

# === 使用 ===
avg_llm_score = compute_avg_llm_score("/data2/liuxj/1-Sentiment-mllm/model_train/result/gold/background.txt", "/data2/liuxj/1-Sentiment-mllm/model_train/baselines/intern_result/task3.txt")
print(f"Average LLM score: {avg_llm_score:.4f}")
