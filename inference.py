from config import TrainConfig
from SentimentAnalyzer import App

if __name__ == "__main__":
    config = TrainConfig()
    config.parse()
    if config.config_file:
        _, config = TrainConfig.import_json(path=config.config_file)

    app = App(config=config, device="cuda")

    input_str = input("input: ")
    output_res = app(batch=[input_str])
    print(output_res)