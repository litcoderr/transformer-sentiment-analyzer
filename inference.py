import numpy as np

from config import TrainConfig
from SentimentAnalyzer import App


if __name__ == "__main__":
    config = TrainConfig()
    config.parse()
    if config.config_file:
        _, config = TrainConfig.import_json(path=config.config_file)

    app = App(config=config, device="cuda")

    while True:
        print("input: ", end="")
        input_str = input()
        if input_str == "":
            break
        output_res = app(batch=[input_str])
        output_res = output_res.detach().cpu().numpy()
        res = np.argmax(output_res) 
        if res == 0:
            print("부정적 댓글")
        else:
            print("긍정적 댓글")