import ast

def load_data(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            id_, content = line.strip().split('\t')
            content_list = ast.literal_eval(content)  # 安全地解析列表格式
            data[id_] = content_list
    return data

def compute_metrics(gold_file, pred_file):
    gold_data = load_data(gold_file)
    pred_data = load_data(pred_file)

    assert gold_data.keys() == pred_data.keys(), "ID不一致！"

    N = len(gold_data)  # 真实样本数
    M = len(pred_data)  # 预测样本数

    precision_scores = []
    recall_scores = []

    for id_ in gold_data:
        gold_list = gold_data[id_]
        pred_list = pred_data[id_]

        gold_dict = {ent.lower(): set([e.lower() for e in emos]) for ent, emos in gold_list}
        pred_dict = {ent.lower(): set([e.lower() for e in emos]) for ent, emos in pred_list}

        # ------- Precision（预测为主） -------
        p_sum = 0
        for ent, pred_emotions in pred_dict.items():
            if ent in gold_dict:
                intersection = pred_emotions & gold_dict[ent]
                emotion_score = len(intersection) / len(pred_emotions) if pred_emotions else 0
                score = 0.5 + 0.5 * emotion_score
            else:
                score = 0  # 实体名不对就不给分
            p_sum += score
        precision_scores.append(p_sum / len(pred_dict) if pred_dict else 0)

        # ------- Recall（真实为主） -------
        r_sum = 0
        for ent, gold_emotions in gold_dict.items():
            if ent in pred_dict:
                intersection = gold_emotions & pred_dict[ent]
                emotion_score = len(intersection) / len(gold_emotions) if gold_emotions else 0
                score = 0.5 + 0.5 * emotion_score
            else:
                score = 0
            r_sum += score
        recall_scores.append(r_sum / len(gold_dict) if gold_dict else 0)

    P_star = sum(precision_scores) / M
    R_star = sum(recall_scores) / N
    F1_star = 2 * P_star * R_star / (P_star + R_star + 1e-8)

    return {"P*": P_star, "R*": R_star, "F1*": F1_star}

# 使用方式
result = compute_metrics("/data2/liuxj/1-Sentiment-mllm/model_train/result/im_em/exist_out_gw.txt", "/data2/liuxj/1-Sentiment-mllm/model_train/result/im_em/exist_out_pw.txt")
print(result)
