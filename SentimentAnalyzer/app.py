import torch
import torch.nn.functional as F
import transformers
from transformers import BertTokenizerFast, EncoderDecoderModel
from typing import List, Union


class App:
    def __init__(self, device="cuda"):
        self.device = device
        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/bertshared-kor-base")

    def __call__(self, batch: List[Union[str, int]], preprocessed=False):
        # preprocess text to tensor
        if not preprocessed:
            tensor = self.preprocess(batch)

        # forward

        # postprocess tensor to text
        pass

    def forward(self, input_tensor: torch.Tensor, is_train=False):
        pass

    def preprocess(self, batch_of_text: List[str]):
        max_length = 0
        tensor_list = []
        for s in batch_of_text:
            # torch.Tensor: [1, token_length]
            input_ids = self.tokenizer.encode(s, return_tensors='pt').to(self.device)
            tensor_list.append(input_ids)

            if max_length < input_ids.shape[1]:
                max_length = input_ids.shape[1]
        
        padded_list = []
        for t in tensor_list:
            padded = F.pad(t, (0,max_length-t.shape[1]), "constant", 0)
            padded_list.append(padded)

        batch_tensor = torch.cat(padded_list, 0) 
        return batch_tensor

    def postprocess(self, batch_of_tensor):
        pass