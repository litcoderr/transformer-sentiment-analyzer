import os
import glob
import shutil

from config import TrainConfig
from SentimentAnalyzer import App

if __name__ == "__main__":
    version_name = "v1.1_mini"
    checkpoint_dir = "./SentimentAnalyzer/checkpoints"
    json_paths = glob.glob(os.path.join(checkpoint_dir, "{}*.json".format(version_name)))
    for json_path in json_paths:
        _, config = TrainConfig.import_json(path=json_path)
        os.remove(config.ckpt_path)
        os.remove(json_path)

    tensorlog_dir = os.path.join("./tensorlog", version_name) 
    shutil.rmtree(tensorlog_dir)