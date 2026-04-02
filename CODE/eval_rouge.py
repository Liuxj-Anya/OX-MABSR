from rouge_score import rouge_scorer

def load_id2text(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            id_, text = line.strip().split('\t', 1)
            data[id_] = text
    return data

def compute_avg_rouge_l(ref_file, pred_file):
    ref_data = load_id2text(ref_file)
    pred_data = load_id2text(pred_file)

    assert ref_data.keys() == pred_data.keys(), "ID 不匹配！"

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []

    for id_ in ref_data:
        ref_text = ref_data[id_]
        pred_text = pred_data[id_]
        score = scorer.score(ref_text, pred_text)['rougeL'].fmeasure
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    return avg_score

# 使用方式
avg_rouge = compute_avg_rouge_l("/data2/liuxj/1-Sentiment-mllm/model_train/result/gold/think.txt", "/data2/liuxj/1-Sentiment-mllm/model_train/baselines/intern_result/task4.txt")
print(f"Average ROUGE-L F1: {avg_rouge:.4f}")
