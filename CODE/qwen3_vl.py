import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model,prepare_model_for_kbit_training
from transformers import PreTrainedModel, PretrainedConfig

class TextImageConfig(PretrainedConfig):
    model_type = "text-image-causal-lm"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_model_path = kwargs.get("text_model_path", "")
        self.image_feature_dim = kwargs.get("image_feature_dim", 1536)

class Qwen3_TextImageModel(PreTrainedModel):
    config_class = TextImageConfig
    def __init__(self, config, lora_config=None,tokenizer=None):
        super().__init__(config)

        self.text_model = AutoModelForCausalLM.from_pretrained(
            config.text_model_path,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer=tokenizer
        self.text_model.resize_token_embeddings(len(self.tokenizer))

        # for p in self.text_model.get_input_embeddings().parameters():
        #     p.requires_grad = False

        if lora_config is not None:
            #self.text_model = prepare_model_for_kbit_training(self.text_model)
            self.text_model = get_peft_model(self.text_model, lora_config)

        # self.text_model.config.hidden_dropout_prob = 0.3
        # self.text_model.config.attention_probs_dropout_prob = 0.1

        image_feature_dim=config.image_feature_dim
        text_embedding_dim = self.text_model.config.hidden_size
        # 图像投影层，并放到与主模型同一设备
        #self.image_projector = nn.Linear(image_feature_dim, text_embedding_dim)
        self.image_projector = nn.Sequential(
            nn.Linear(image_feature_dim, text_embedding_dim),
            nn.ReLU(),
            nn.Linear(text_embedding_dim, text_embedding_dim),
            nn.Dropout(p=0.1)
        )

        # 冻结除特殊 token 外的 embedding
        # 独立 embedding，只处理 <img> 和 </img>
        self.special_token_embed = nn.Embedding(7, text_embedding_dim)  # index 0: <img>, 1: </img>
        # 保存 token id 映射
        self.img_token_id = self.tokenizer.convert_tokens_to_ids('<img>')
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids('</img>')
        with torch.no_grad():
            self.special_token_embed.weight[0] = self.text_model.get_input_embeddings().weight[self.img_token_id]
            self.special_token_embed.weight[1] = self.text_model.get_input_embeddings().weight[self.img_end_token_id]

    def _move_to_first_device(self, module):
        if hasattr(self.text_model, 'hf_device_map'):
            first_module_name = list(self.text_model.hf_device_map.keys())[0]
            target_device = self.text_model.hf_device_map[first_module_name]
            return module.to(target_device)
        else:
            return module.to('cuda' if torch.cuda.is_available() else 'cpu')

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        """
        为了兼容 Trainer 的调用，重载 generate，使其内部自动调用 generate_with_image
        """
        image_features = kwargs.get("image_features")
        task_id = kwargs.get("task_id", 6)  # 默认任务 ID 为 6

        if image_features is None:
            raise ValueError("generate() 需要 image_features")

        generation_kwargs = {
            "max_new_tokens": 3000,
            # "do_sample": False,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "temperature": 0.7,
            "num_return_sequences": 2,
            "do_sample": True
        }

        return self.generate_with_image(
            input_ids=input_ids,
            image_features=image_features,
            attention_mask=attention_mask,
            generation_kwargs=generation_kwargs,
            task_id=task_id
        )

    def generate_with_image(self, input_ids, image_features, attention_mask=None, generation_kwargs=None,task_id=None):
        self.eval()

        generation_kwargs = generation_kwargs or {}

        # 清除 transformers.generate 不支持的 kwargs
        generation_kwargs.pop("image_features", None)
        generation_kwargs.pop("task_id", None)

        device = self.image_projector[0].weight.device

        self.special_token_embed.to(device)

        input_ids = input_ids.to(device)

        proj_dtype = next(self.image_projector.parameters()).dtype
        proj_device = next(self.image_projector.parameters()).device
        image_features = image_features.to(device=proj_device, dtype=proj_dtype)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 投影图像特征
        projected_image_features = self.image_projector(image_features)  # [B, N, hidden]

        B = projected_image_features.size(0)
        token_ids = torch.tensor([0, 1],device=device)  # 0: <img>, 1: </img>
        img_token_embed, img_end_token_embed = self.special_token_embed(token_ids).unbind(0)
        img_token_embed = img_token_embed.unsqueeze(0).expand(B, 1, -1)
        img_end_token_embed = img_end_token_embed.unsqueeze(0).expand(B, 1, -1)
        task_id_token_embed = self.special_token_embed(torch.tensor([task_id],device=device)).unsqueeze(0).expand(B, 1, -1)

        # print(projected_image_features.shape)
        # print(task_id_token_embed.shape)

        image_input_embeds = torch.cat([img_token_embed, projected_image_features, img_end_token_embed,task_id_token_embed], dim=1)

        # 获取文本 embedding
        text_input_embeds = self.text_model.get_input_embeddings()(input_ids)  # [B, T, hidden]

        # 拼接图像 + 文本 embedding
        inputs_embeds = torch.cat([image_input_embeds, text_input_embeds], dim=1)

        inputs_embeds = inputs_embeds.to(dtype=self.text_model.dtype)

        # 拼接 attention_mask
        if attention_mask is not None:
            image_mask = torch.ones((B, image_input_embeds.size(1)), dtype=attention_mask.dtype).to(device)
            attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        # 使用 generate（注意我们传入的是 inputs_embeds）
        outputs = self.text_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs
        )

        return outputs

    def forward(self, input_ids, attention_mask=None, labels=None, image_features=None, task_id=None,
                logits_to_keep=None):

        proj_dtype = next(self.image_projector.parameters()).dtype
        proj_device = next(self.image_projector.parameters()).device
        image_features = image_features.to(device=proj_device, dtype=proj_dtype)
        projected_image_features = self.image_projector(image_features)

        B = projected_image_features.size(0)

        # 获取 embedding 的设备和 dtype
        embed_device = self.special_token_embed.weight.device
        task_id = task_id.to(device=embed_device, dtype=torch.long)

        # 获取特殊 token 的 embedding（image start, end, task_id）
        img_token_embed = self.special_token_embed(torch.tensor([0], device=embed_device)).unsqueeze(0).expand(B, 1, -1)
        img_end_token_embed = self.special_token_embed(torch.tensor([1], device=embed_device)).unsqueeze(0).expand(B, 1, -1)
        task_id_token_embed = self.special_token_embed(task_id).unsqueeze(1)  # (B, 1, D)

        
        # 拼接 image 特征 + 特殊 token
        image_input_embeds = torch.cat(
            [img_token_embed, projected_image_features, img_end_token_embed, task_id_token_embed], dim=1)

        embed_device = self.text_model.get_input_embeddings().weight.device
        input_ids = input_ids.to(embed_device)
        attention_mask = attention_mask.to(embed_device)
        inputs_embeds = self.text_model.get_input_embeddings()(input_ids)  # [B, seq_len, hidden]

        # 拼接图像和文本部分
        combined_inputs_embeds = torch.cat([image_input_embeds, inputs_embeds], dim=1)

        # 拼接 attention_mask
        if attention_mask is not None:
            image_len = image_input_embeds.size(1)
            image_mask = torch.ones((B, image_len), dtype=attention_mask.dtype, device=embed_device)
            attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        # 拼接 labels
        if labels is not None:
            image_labels = torch.full(
                (labels.size(0), image_input_embeds.size(1)),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            labels = torch.cat([image_labels, labels], dim=1)

        # 前向传播
        outputs = self.text_model(
            inputs_embeds=combined_inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        # 如果是 RL 模式（无 labels），支持裁剪 logits
        if logits_to_keep is not None and labels is None:
            outputs.logits = outputs.logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, vocab_size)

        return outputs

    # def forward(self, input_ids, attention_mask=None, labels=None,image_features=None,task_id=None):
    #
    #     # [B, image_tokens, 2048] → [B, image_tokens, hidden]
    #     device = input_ids.device
    #     projected_image_features = self.image_projector(image_features)
    #
    #     B = projected_image_features.size(0)
    #     # 用独立的 embedding 生成特殊 token embedding
    #     img_token_embed = self.special_token_embed(torch.tensor([0], device=device)).unsqueeze(0).expand(B, 1, -1)
    #     img_end_token_embed = self.special_token_embed(torch.tensor([1], device=device)).unsqueeze(0).expand(B, 1, -1)
    #
    #     task_id_token_embed = self.special_token_embed(task_id).unsqueeze(0).expand(B, 1, -1)
    #
    #     image_input_embeds = torch.cat([img_token_embed, projected_image_features, img_end_token_embed,task_id_token_embed], dim=1)
    #
    #     inputs_embeds = self.text_model.get_input_embeddings()(input_ids)  # [B, seq_len, hidden]
    #
    #     # 拼接图像+文本
    #     combined_inputs_embeds = torch.cat([image_input_embeds, inputs_embeds], dim=1)
    #
    #     # attention_mask 拼接
    #     if attention_mask is not None:
    #         B = attention_mask.size(0)
    #         image_len = image_input_embeds.size(1)
    #         image_mask = torch.ones((B, image_len), dtype=attention_mask.dtype).to(attention_mask.device)
    #         attention_mask = torch.cat([image_mask, attention_mask], dim=1)
    #
    #     # labels 拼接（用 -100 表示图像部分不计算loss）
    #     if labels is not None:
    #         image_labels = torch.full(
    #             (labels.size(0), image_input_embeds.size(1)),
    #             -100,
    #             dtype=labels.dtype,
    #             device=labels.device
    #         )
    #         labels = torch.cat([image_labels, labels], dim=1)
    #
    #     outputs = self.text_model(
    #         inputs_embeds=combined_inputs_embeds,
    #         attention_mask=attention_mask,
    #         labels=labels
    #     )
    #
    #     return outputs  # 包含 loss 和 logits



