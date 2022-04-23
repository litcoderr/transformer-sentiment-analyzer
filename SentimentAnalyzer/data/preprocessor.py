import os
import urllib.request as request

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(CURRENT_DIR, "dataset")

def download_data():
    train_data_name = os.path.join(DATASET_DIR, "train.txt")
    if not os.path.exists(train_data_name):
        print("downloading trian data...")
        request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
                            filename=train_data_name)
    else:
        print("train data exists")

    test_data_name = os.path.join(DATASET_DIR, "test.txt")
    if not os.path.exists(test_data_name):
        print("downloading test data...")
        request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
                            filename=test_data_name)
    else:
        print("test data exists")