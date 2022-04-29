import torch

class App:
    def __init__(self, device="cuda"):
        self.device = device
        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/bertshared-kor-base")

    def __call__(self, text):
        # preprocess text to tensor

        # forward

        # postprocess tensor to text
        pass

    def forward(self, input_tensor: torch.Tensor, is_train=False):
        pass

    def preprocess(self, batch_of_text):
        pass

    def postprocess(self, batch_of_tensor):
        pass