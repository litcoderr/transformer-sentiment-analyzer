import os
import torch
import torch.nn.functional as F
import transformers
from transformers import BertTokenizerFast, EncoderDecoderModel
from typing import List, Union

from .model.transformer import ModelConfig, TransformerV1, TransformerV1_1, TransformerV1_residual

MODEL_VERSION = {
    "v1.0": TransformerV1, # mean of all encoder output
    "v1.1": TransformerV1_1, # lr 0.001, use last encoder output
    "v1.11": TransformerV1_1, # lr 0.0001, use last encoder output
    "v1_mini": TransformerV1_1, # lr 0.0001, use last encoder output, minified version for easier training
    "v1.1_mini": TransformerV1_1, # lr 0.0001, use last encoder output, minified version for easier training
    "v1_res": TransformerV1_residual # lr 0.0001, use last encoder output, minified version for easier training
}

class App:
    def __init__(self, config, device="cuda"):
        self.model_config = config.model_config # model configuration
        self.device = device

        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/bertshared-kor-base")

        self.model = MODEL_VERSION[config.version](config=self.model_config, device=self.device)
        self.model = self.model.to(self.device)

        # load checkpoint if exists
        if os.path.exists(config.ckpt_path):
            checkpoint = torch.load(config.ckpt_path)
            self.model.load_state_dict(checkpoint["checkpoint"])
            print("loaded {}...".format(config.ckpt_path))

    def __call__(self, batch: List[str]):
        # preprocess text to tensor
        tensor = self.preprocess(batch)

        # forward
        fwd_res = self.forward(tensor, is_train=False)

        # TODO postprocess tensor to text
        return fwd_res

    def forward(self, input_tensor: torch.Tensor, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        input_tensor = input_tensor.to(self.device)
        output_tensor = self.model(input_tensor)
        return output_tensor

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