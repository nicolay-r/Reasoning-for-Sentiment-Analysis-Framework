import os
import shutil
from os.path import join

from src.ft.service import THoRFrameworkService
from src.service_csv import CsvService
from utils import DATA_DIR, LABEL_MAP


def convert_rusentne2023_dataset(src, target, has_label=True):
    print(f"Reading source: {src}")
    cols = ["sentence", "entity"] + (["label"] if has_label else [])
    records_it = [[item[0], item[1]] + [int(item[2]) if has_label else 0]
                  for item in CsvService.read(target=src, skip_header=True, delimiter="\t", cols=cols)]
    THoRFrameworkService.write_dataset(target_template=target, entries_it=records_it, label_map=LABEL_MAP)


if __name__ == "__main__":

    t_dir = join(DATA_DIR, "rusentne2023")

    data = {
        # Data related to RuSentNE competitions.
        join(t_dir, "train_en.csv"): join(DATA_DIR, "train_data_en.csv"),
        join(t_dir, "valid_en.csv"): join(DATA_DIR, "valid_data_en.csv"),
        join(t_dir, "test_en.csv"): join(DATA_DIR, "final_data_en.csv"),
    }

    pickle_rusentne2023_data = {
        join(t_dir, "Rusentne2023_train"): join(t_dir, "train_en.csv"),
        join(t_dir, "Rusentne2023_valid"): join(t_dir, "valid_en.csv"),
        join(t_dir, "Rusentne2023_test"): join(t_dir, "test_en.csv"),
    }

    for d in [DATA_DIR, t_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    for target, source in data.items():
        shutil.copy(source, target)

    for target, source in pickle_rusentne2023_data.items():
        convert_rusentne2023_dataset(source, target, has_label=not "test" in target)
