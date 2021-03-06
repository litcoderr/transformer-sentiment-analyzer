import os
import pickle
from tqdm import tqdm
from array import array

from SentimentAnalyzer.app import App
from .dataset import read_raw, DATA_PATH
from config import TrainConfig

if __name__ == "__main__":
    config = TrainConfig()
    config.parse()
    if config.config_file:
        _, config = TrainConfig.import_json(path=config.config_file)
    app = App(config=config)

    # preprocess
    for typ in DATA_PATH.keys():
        raw = read_raw(DATA_PATH[typ]["raw"])
        pbar = tqdm(raw)
        for ids, comment, label in pbar:
            pbar.set_description("Processing {}...".format(typ))

            input_tensor = app.preprocess([comment]).to("cpu")
            preprocessed = input_tensor.tolist()
            preprocessed.append(int(label))
            preprocessed = tuple(preprocessed)

            # save pickle
            pickle_path = os.path.join(DATA_PATH[typ]["processed"], ids)
            with open(pickle_path, "wb") as file:
                pickle.dump(preprocessed, file) 