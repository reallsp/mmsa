import torch
import torch.nn as nn
import os
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

__all__ = ['BertTextEncoder']

TRANSFORMERS_MAP = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
}

class BertTextEncoder(nn.Module):
    def __init__(
        self,
        use_finetune: bool = False,
        transformers: str = 'bert',
        pretrained: str = 'bert-base-uncased',
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
    ):
        super().__init__()

        tokenizer_class = TRANSFORMERS_MAP[transformers][1]
        model_class = TRANSFORMERS_MAP[transformers][0]

        # 如果给的是本地目录/文件，则默认强制离线加载，避免在无外网环境下 from_pretrained 超时。
        if local_files_only is None:
            local_files_only = os.path.exists(pretrained)

        self.tokenizer = tokenizer_class.from_pretrained(
            pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.model = model_class.from_pretrained(
            pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.use_finetune = use_finetune
    
    def get_tokenizer(self):
        return self.tokenizer
    
    # def from_text(self, text):
    #     """
    #     text: raw data
    #     """
    #     input_ids = self.get_id(text)
    #     with torch.no_grad():
    #         last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
    #     return last_hidden_states.squeeze()
    
    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
