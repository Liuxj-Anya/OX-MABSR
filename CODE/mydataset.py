import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np
import random

class MyDataset(Dataset):
    def __init__(self, data_path: str, tokenizer):

        self.tokenizer = tokenizer
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = f.readlines()

    def __len__(self):
        return len(self.samples)

    def data_process(self,data):

        id,answer = data.strip().split('\t')
        with open('data/text/'+id+'.txt','r',encoding='utf-8') as ft:
            text = ft.readline().strip()
        image_array = np.load('data/image_feature/'+id+'.npy')
        input_text = "Given an image and a sentence: [" + text + "] , Please think step by step, obtain the sentiments and causes of the entities from multiple levels and angles, and finally output the list of entities and sentiments."

        return image_array,input_text,answer

    def __getitem__(self, idx):

        image_tensor,input_text, target_text = self.data_process(self.samples[idx])

        input_enc = self.tokenizer(
            f"<s><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors="pt"
            )
        response = self.tokenizer(f"{target_text+self.tokenizer.eos_token}",
                                  add_special_tokens=False,
                                  padding=False,
                                  truncation=True,
                                  return_tensors="pt"
                                  )

        # 拼接 input_ids, attention_mask, labels
        input_ids = torch.cat([input_enc["input_ids"], response["input_ids"]], dim=1).squeeze(0)
        attention_mask = torch.cat([input_enc["attention_mask"], response["attention_mask"]], dim=1).squeeze(0)

        # 构建 labels，input部分填充 -100，response部分保留
        labels = torch.cat([
            torch.full_like(input_enc["input_ids"], -100),
            response["input_ids"]
        ], dim=1).squeeze(0)

        task_id = 6

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image_features": image_tensor,  # 示例用法,
            "task_id": task_id
        }

class GrpoDataset(Dataset):
    def __init__(self, data_path: str, tokenizer):

        self.tokenizer = tokenizer
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = f.readlines()

    def __len__(self):
        return len(self.samples)

    def data_process(self,data):

        id,answer = data.strip().split('\t')
        with open('data/text/'+id+'.txt','r',encoding='utf-8') as ft:
            text = ft.readline().strip()
        image_array = np.load('data/image_feature/'+id+'.npy')
        input_text = "Given an image and a sentence: [" + text + "] , Please think step by step, obtain the sentiments and causes of the entities from multiple levels and angles, and finally output the list of entities and sentiments."

        return image_array,input_text,answer

    def __getitem__(self, idx):

        image_tensor,input_text, target_text = self.data_process(self.samples[idx])

        full_prompt = (
            f"<s><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

        task_id = 6

        return {
            "prompt": full_prompt,  # full string with system+user prompt
            "target_text": target_text,  # response
            "image_features": image_tensor,
            "task_id": task_id
        }

class MultiTaskDataset(Dataset):
    def __init__(self, task_paths: dict, tokenizer):
        """
        task_paths: dict，例如：
            {
                "task1": "data/task1.txt",
                "task2": "data/task2.txt",
                "task3": "data/task3.txt"
                "task4": "data/task4.txt"
            }
        """
        self.tokenizer = tokenizer
        self.task_file_handles = {}
        self.task_line_offsets = {}
        self.task_names = list(task_paths.keys())

        # 为每个任务文件记录每行的 offset，避免一次性读入全部
        for task, path in task_paths.items():
            self.task_file_handles[task] = open(path, 'r', encoding='utf-8')
            self.task_line_offsets[task] = self._build_line_offsets(path)

        # 计算总样本数用于 __len__
        self.total_len = sum(len(v) for v in self.task_line_offsets.values())

        # 所有任务混合的样本索引： (task_name, line_index)
        self.index_map = []
        for task, offsets in self.task_line_offsets.items():
            self.index_map.extend([(task, i) for i in range(len(offsets))])
        random.shuffle(self.index_map)  # 混合任务

    def _build_line_offsets(self, filepath):
        offsets = []
        offset = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                offsets.append(offset)
                offset += len(line.encode('utf-8'))
        return offsets

    def __len__(self):
        return self.total_len

    def data_process(self, task_name, line):
        if task_name == "task1":
            id,answer,text = line.strip().split('\t')
            image_array = np.load(f'data/twitter_image/{id}.npy')
            input_text = "Extract Explicit entity and its Simple Sentiment(POS or NEU or NEG): Given an image and a sentence: [" + text + "] , please identify the entities and sentiment in the sentence. Output the results in the following format: (Entity, Simple Sentiment)."
            task_id=2
        elif task_name == "task2":
            #print(line)
            id, answer = line.strip().split('\t')
            with open(f'data/text/{id}.txt', 'r', encoding='utf-8') as ft:
                text = ft.readline().strip()
            image_array = np.load(f'data/image_feature/{id}.npy')
            input_text = "Extract Implicit entity and its Complex Sentiment: Given an image and a sentence: ["+text+"] , please identify the entities and sentiment of the historical event associated with this image and sentence. Output the results in the following format: (Entity, Complex Sentiment)."
            task_id = 3
        elif task_name == "task3":
            id, answer = line.strip().split('\t')
            with open(f'data/text/{id}.txt', 'r', encoding='utf-8') as ft:
                text = ft.readline().strip()
            image_array = np.load(f'data/image_feature/{id}.npy')
            input_text = "Shallow reasoning: Given an image and a sentence: ["+text+"] , please use the aesthetic information of the image, the facial expressions of the characters in the image, and the scene information of the image to judge the entity and its sentiment, and explain the causes for this sentiment."
            task_id = 4
        elif task_name == "task4":
            id, answer = line.strip().split('\t')
            with open(f'data/text/{id}.txt', 'r', encoding='utf-8') as ft:
                text = ft.readline().strip()
            image_array = np.load(f'data/image_feature/{id}.npy')
            input_text = "Deep reasoning: Given an image and a sentence: ["+text+"] , please use the historical context, important events,and relevant background information of the image to judge the entity and its sentiment, and explain the causes for this sentiment."
            task_id = 5
        elif task_name == "task5":
            id, answer = line.strip().split('\t')
            with open(f'data/text/{id}.txt', 'r', encoding='utf-8') as ft:
                text = ft.readline().strip()
            image_array = np.load(f'data/image_feature/{id}.npy')
            #input_text = "Given an image and a sentence: [" + text + "] , Please think step by step, explain the reasons from multiple levels and angles, and obtain the entities and sentiments of the events related to the image and sentence."
            input_text = "Given an image and a sentence: [" + text + "] , Please think step by step, obtain the sentiments and causes of the entities from multiple levels and angles, and finally output the list of entities and sentiments."

            task_id = 6
        else:
            raise ValueError(f"Unknown task: {task_name}")

        return image_array, input_text, answer,task_id

    def __getitem__(self, idx):
        task_name, line_index = self.index_map[idx]
        offset = self.task_line_offsets[task_name][line_index]
        f = self.task_file_handles[task_name]
        f.seek(offset)
        line = f.readline()

        image_tensor, input_text, target_text,task_id = self.data_process(task_name, line)

        input_enc = self.tokenizer(
            f"<s><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        response = self.tokenizer(f"{target_text + self.tokenizer.eos_token}",
                                  add_special_tokens=False,
                                  padding=False,
                                  truncation=True,
                                  return_tensors="pt"
                                  )

        # 拼接 input_ids, attention_mask, labels
        input_ids = torch.cat([input_enc["input_ids"], response["input_ids"]], dim=1).squeeze(0)
        attention_mask = torch.cat([input_enc["attention_mask"], response["attention_mask"]], dim=1).squeeze(0)

        # 构建 labels，input部分填充 -100，response部分保留
        labels = torch.cat([
            torch.full_like(input_enc["input_ids"], -100),
            response["input_ids"]
        ], dim=1).squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image_features": image_tensor,  # 示例用法
            "task_id": task_id
        }


