import importlib
import os
import sys
from collections import Counter

from tqdm import tqdm

import requests


def download(dest_file_path, source_url):
    print(('Downloading from {src} to {dest}'.format(src=source_url, dest=dest_file_path)))

    sys.stdout.flush()
    datapath = os.path.dirname(dest_file_path)

    if not os.path.exists(datapath):
        os.makedirs(datapath, mode=0o755)

    dest_file_path = os.path.abspath(dest_file_path)

    r = requests.get(source_url, stream=True)
    total_length = int(r.headers.get('content-length', 0))

    with open(dest_file_path, 'wb') as f:
        pbar = tqdm(total=total_length, unit='B', unit_scale=True)
        for chunk in r.iter_content(chunk_size=32 * 1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)


def iter_with_ids(content_it):
    for item_id, item in enumerate(content_it):
        yield item_id, item


def find_by_prefix(d, key):
    """
        d: dict (str, val).
    """
    assert(isinstance(d, dict))
    assert(isinstance(key, str))

    # We first check the full match.
    for k, value in d.items():
        if k == key:
            return value

    # If we can't establish full match, then we seek by prefix.
    matches = []
    for k, value in d.items():
        if key.startswith(k):
            matches.append(k)

    if len(matches) > 1:
        raise Exception(f"There are multiple entries that are related to `{key}`: {matches}")
    if len(matches) == 0:
        raise Exception(f"No entries were found for `{key}`!")

    return d[matches[0]]


def iter_params(text):
    assert(isinstance(text, str))
    beg = 0
    while beg < len(text):
        try:
            pb = text.index('{', beg)
        except ValueError:
            break
        pe = text.index('}', beg+1)
        # Yield argument.
        yield text[pb+1:pe]
        beg = pe+1


def format_model_name(name):
    return name.replace("/", "_")


def parse_filepath(filepath, default_filepath=None, default_ext=None):
    """ This is an auxiliary function for handling sources and targets from cmd string.
    """
    if filepath is None:
        return default_filepath, default_ext, None
    info = filepath.split(":")
    filepath = info[0]
    meta = info[1] if len(info) > 1 else None
    ext = filepath.split('.')[-1] if default_ext is None else default_ext
    return filepath, ext, meta


def handle_table_name(name):
    return name.\
        replace('-', '_').\
        replace('.', "_")


def auto_import(name, is_class=False):
    """ Import from the external python packages.
    """
    def __get_module(comps_list):
        return importlib.import_module(".".join(comps_list))

    components = name.split('.')
    m = getattr(__get_module(components[:-1]), components[-1])

    return m() if is_class else m


def optional_limit_iter(it_data, limit=None):
    counter = Counter()
    for data in it_data:
        counter["returned"] += 1
        if limit is not None and counter["returned"] > limit:
            break
        yield data
