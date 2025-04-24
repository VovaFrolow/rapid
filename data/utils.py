import math
import os
import pathlib
import re
import urllib.parse

import requests
import tqdm
import errno
import gdown


def get_default_dir(subdir):
    try:
        import videosaur.data

        default_dir = videosaur.data.get_data_root_dir()
    except ImportError:
        default_dir = "./data"
    default_dir = "/code/datasets/coco_raw"
    
    return os.path.join(default_dir, subdir)

def get_shard_pattern(path: str):
    base_pattern: str = "shard-%06d.tar"
    return os.path.join(path, base_pattern)

def download_file(url: str, dest_dir: str) -> pathlib.Path:
    """Download file to location and return path to location.

    Adapted from https://stackoverflow.com/a/10744565 and https://stackoverflow.com/a/53299682.
    """
    response = requests.get(url, stream=True)

    if "Content-Disposition" in response.headers:
        file_name = re.findall("filename=(.+)", response.headers["Content-Disposition"])[0]
    else:
        url_path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
        file_name = pathlib.Path(url_path).name

    if "Content-Length" in response.headers:
        file_size_kb = math.ceil(int(response.headers["Content-Length"]) / 1024)
    else:
        file_size_kb = None

    dest_file = pathlib.Path(dest_dir) / (file_name + ".tmp")
    with open(dest_file, "wb") as handle:
        for data in tqdm.tqdm(response.iter_content(chunk_size=1024), unit="kB", total=file_size_kb):
            handle.write(data)

    dest_file = dest_file.rename(dest_file.with_name(file_name))

    return dest_file

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

CHUNK_SIZE = 32768

def download_file_from_google_drive(url, destination):
    gdown.download(url, destination, quiet=False)