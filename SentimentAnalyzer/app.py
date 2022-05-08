import torch
import torch.nn.functional as F
import transformers
from transformers import BertTokenizerFast, EncoderDecoderModel
from typing import List, Union

from .model.transformer import ModelConfig, Transformer


class App:
    def __init__(self, model_config: ModelConfig, checkpoint=None, device="cuda"):
        self.model_config = model_config # model configuration
        self.checkpoint = checkpoint # checkpoint file path
        self.device = device

        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/bertshared-kor-base")

        self.model = Transformer(config=model_config)
        self.model = self.model.to(self.device)

        # TODO load checkpoint if exists

    def __call__(self, batch: List[str]):
        # preprocess text to tensor
        tensor = self.preprocess(batch)

        # forward
        fwd_res = self.forward(tensor, is_train=False)

        # TODO postprocess tensor to text
        pass

    def forward(self, input_tensor: torch.Tensor, is_train=True):
        input_tensor = input_tensor.to(self.device)
        output_tensor = self.model(input_tensor)

    def preprocess(self, batch_of_text: List[str]):
        max_length = 0
        tensor_list = []
        for s in batch_of_text:
            # torch.Tensor: [1, token_length]
            input_ids = self.tokenizer.encode(s, return_tensors='pt').to(self.device, dtype=torch.int32)
            # TODO typecast to integer
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
        # TODO implement postprocess
        pass