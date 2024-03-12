import os
from os.path import join

from src.utils import download
from utils import DATA_DIR


if __name__ == '__main__':

    data = {
        join(DATA_DIR, "train_data.csv"): "https://raw.githubusercontent.com/dialogue-evaluation/RuSentNE-evaluation/main/train_data.csv",
        join(DATA_DIR, "train_data_en.csv"): "https://www.dropbox.com/scl/fi/szj5j87f6w3ershnfh39x/train_data_en.csv?rlkey=h6ve617kl3o8g57otbt3yzamv&dl=1",
        join(DATA_DIR, "valid_data.csv"): "https://www.dropbox.com/scl/fi/vthocgamkyxhqejyyw7hu/validation_data.csv?rlkey=dh3nk108vnfqrl8t3ituaq96o&dl=1",
        join(DATA_DIR, "valid_data_en.csv"): "https://www.dropbox.com/scl/fi/8cmocj7hqbbxpw8bop4vs/valid_data_en.csv?rlkey=6lqaz15lxs5x9hxemuhfv0fnk&dl=1",
        # Resources.
        join(DATA_DIR, "final_data.csv"): "https://raw.githubusercontent.com/dialogue-evaluation/RuSentNE-evaluation/main/final_data.csv",
        join(DATA_DIR, "final_data_en.csv"): "https://www.dropbox.com/scl/fi/rsxym3t611iunvbdgja3m/final_data_en.csv?rlkey=fl6vo81l5qvf3atel5dj88od1&dl=1",
    }

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for target, url in data.items():
        download(dest_file_path=target, source_url=url)
