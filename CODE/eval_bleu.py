import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 添加 punkt 的路径（指向的是 tokenizers/punkt 的上一层）
nltk.data.path.append('/data2/liuxj/1-Sentiment-mllm/model_train')

def load_id2text(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            id_, text = line.strip().split('\t', 1)
            data[id_] = text
    return data

def compute_avg_bleu(ref_file, pred_file):
    ref_data = load_id2text(ref_file)
    pred_data = load_id2text(pred_file)

    assert ref_data.keys() == pred_data.keys(), "ID 不匹配！"

    smoothie = SmoothingFunction().method3
    scores = []

    for id_ in ref_data:
        ref_sent = nltk.word_tokenize(ref_data[id_].lower())
        pred_sent = nltk.word_tokenize(pred_data[id_].lower())

        score = sentence_bleu([ref_sent], pred_sent, weights=(1.0,), smoothing_function=smoothie)
        scores.append(score)

    avg_bleu = sum(scores) / len(scores)
    return avg_bleu

# 使用
avg_score = compute_avg_bleu("/data2/liuxj/1-Sentiment-mllm/model_train/result/gold/think.txt", "/data2/liuxj/1-Sentiment-mllm/model_train/baselines/intern_result/task4.txt")
print(f"Average BLEU score: {avg_score:.4f}")
