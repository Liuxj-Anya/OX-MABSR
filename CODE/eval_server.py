# eval_server.py
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import warnings

app = Flask(__name__)

# 加载 tokenizer
eval_tokenizer = AutoTokenizer.from_pretrained("/data2/liuxj/1-Sentiment-mllm/Qwen/Qwen3-32B", trust_remote_code=True)

# 加载模型到 GPU:3（仅评估用，资源较小）
eval_model = AutoModelForCausalLM.from_pretrained(
    "/data2/liuxj/1-Sentiment-mllm/Qwen/Qwen3-32B",
    device_map={"": 0},  # 将整个模型加载到 GPU:3
    torch_dtype="auto",
    trust_remote_code=True
)

# 构建评估用 pipeline
llm = pipeline("text-generation", model=eval_model, tokenizer=eval_tokenizer)

def get_llm_score(pred, ref):
    prompt = f"""Given the following gold reasoning path and predicted reasoning path, score the prediction's coherence and emotional consistency compared to the gold one. Gold: {ref} Prediction: {pred} Return only a number between 0 and 1 (inclusive). Do not include any explanation or extra text. Just output the number. Score:"""
    try:
        output = llm(prompt, max_new_tokens=5, return_full_text=False)[0]["generated_text"]
        match = re.search(r"\d*\.?\d+", output)
        if match:
            score = float(match.group())
            return min(max(score, 0.0), 1.0)
        else:
            warnings.warn(f"[get_llm_score] No valid score found in output: {output}")
            return 1.0
    except Exception as e:
        warnings.warn(f"[get_llm_score] Exception during scoring: {e}")
        return 1.0

@app.route("/score", methods=["POST"])
def score():
    data = request.json
    pred = data["pred"]
    ref = data["ref"]
    score = get_llm_score(pred, ref)
    return jsonify({"score": score})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
