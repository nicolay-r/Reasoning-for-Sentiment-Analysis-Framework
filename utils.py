from os.path import dirname, realpath, join

current_dir = dirname(realpath(__file__))
DATA_DIR = join(current_dir, "data")

LABEL_MAP = {1: 1, 0: 0, -1: 2}
LABEL_MAP_REVERSE = {v: k for k, v in LABEL_MAP.items()}
