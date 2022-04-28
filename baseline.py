import transformers
from transformers import BertTokenizerFast, EncoderDecoderModel

transformers.logging.set_verbosity_error()

class Summerize():
    def __init__(self, model_name):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = EncoderDecoderModel.from_pretrained(model_name)
    
    def __call__(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors='pt')

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
    res = summerizer("""안녕. 내 이름은 지영채이고, 나는 서울로 최근에 이사왔어.
    다들 처음 보는데 잘 지냈으면 좋겠어. 나는 그리고 컴퓨터 공학을 전공하고 있단다.
    컴퓨터와 내가 좋아하는 음악 등으로 재밌는 일들을 하며 살고 싶어.
    반갑다. 
    """)

    print(res)