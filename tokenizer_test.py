import time
import transformers
from transformers import BertTokenizerFast, EncoderDecoderModel

transformers.logging.set_verbosity_error()

class Summerize():
    def __init__(self, model_name):
        self.device = "cuda:0"
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = EncoderDecoderModel.from_pretrained(model_name).to(self.device)
    
    def __call__(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)

        sentence_length = len(input_ids[0])
        min_length = max(10, int(0.1*sentence_length))
        max_length = min(128, int(0.3*sentence_length))

        outputs = self.model.generate(
            input_ids,
            min_length=min_length,
            max_length=max_length
        )

        res_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return res_text
    

if __name__ == "__main__":
    # use korean bert baseline
    summerizer = Summerize(model_name="kykim/bertshared-kor-base")
    res = summerizer("""
    학교 주관 비룡제, 체육대회, 비룡 학술회 등과 기숙사 주관 신입생 환영회, 비룡나이트, 광란의 밤 등이 있다.
    비룡제란 여느 학교의 축제와 같은 것이며 비룡 학술대회는 공연과 같은 것을 하는 행사가 아닌 재학생들이 1년동안 공부,
    연구해왔던 것들을 발표하는 행사이다. 2022년에는 당일 오전에 비룡제가 끝난 후 오후에 대전과고와 롤 대전이
    있을 예정이다. 원래 대전과고와 체육대회 공동 개최하려 했으나 코로나 때문에 취소되었다. 공부는 져도 롤은 이겨야 하지 않겠는가.
    """)

    print(res)