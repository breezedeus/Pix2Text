# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).

import hashlib
import os
from pathlib import Path
import logging
import platform
import zipfile
import requests
from typing import Union

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch import Tensor
from torchvision.utils import save_image


fmt = '[%(levelname)s %(asctime)s %(funcName)s:%(lineno)d] %(' 'message)s '
logging.basicConfig(format=fmt)
logging.captureWarnings(True)
logger = logging.getLogger()


def set_logger(log_file=None, log_level=logging.INFO, log_file_level=logging.NOTSET):
    """
    Example:
        >>> set_logger(log_file)
        >>> logger.info("abc'")
    """
    log_format = logging.Formatter(fmt)
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        if not Path(log_file).parent.exists():
            os.makedirs(Path(log_file).parent)
        if isinstance(log_file, Path):
            log_file = str(log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def check_context(context):
    if isinstance(context, str):
        return any([ctx in context.lower() for ctx in ('gpu', 'cpu', 'cuda')])
    if isinstance(context, list):
        if len(context) < 1:
            return False
        return all(isinstance(ctx, torch.device) for ctx in context)
    return isinstance(context, torch.device)


def data_dir_default():
    """

    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'pix2text')
    else:
        return os.path.join(os.path.expanduser("~"), '.pix2text')


def data_dir():
    """

    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('PIX2TEXT_HOME', data_dir_default())


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]


def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if (
        overwrite
        or not os.path.exists(fname)
        or (sha1_hash and not check_sha1(fname, sha1_hash))
    ):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        logger.info('Downloading %s from %s...' % (fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None:  # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(
                    r.iter_content(chunk_size=1024),
                    total=int(total_length / 1024.0 + 0.5),
                    unit='KB',
                    unit_scale=False,
                    dynamic_ncols=True,
                ):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning(
                'File {} is downloaded but the content hash does not match. '
                'The repo may be outdated or download may be incomplete. '
                'If the "repo_url" is overridden, consider switching to '
                'the default repo.'.format(fname)
            )

    return fname


def get_model_file(url, model_dir):
    r"""Return location for the downloaded models on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    url: str
    model_dir : str, default $CNOCR_HOME
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    model_dir = os.path.expanduser(model_dir)
    par_dir = os.path.dirname(model_dir)
    os.makedirs(par_dir, exist_ok=True)

    zip_file_path = os.path.join(par_dir, os.path.basename(url))
    if not os.path.exists(zip_file_path):
        download(url, path=zip_file_path, overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(par_dir)
    os.remove(zip_file_path)

    return model_dir


def read_tsv_file(fp, sep='\t', img_folder=None, mode='eval'):
    img_fp_list, labels_list = [], []
    num_fields = 2 if mode != 'test' else 1
    with open(fp) as f:
        for line in f:
            fields = line.strip('\n').split(sep)
            assert len(fields) == num_fields
            img_fp = (
                os.path.join(img_folder, fields[0])
                if img_folder is not None
                else fields[0]
            )
            img_fp_list.append(img_fp)

            if mode != 'test':
                labels = fields[1].split(' ')
                labels_list.append(labels)

    return (img_fp_list, labels_list) if mode != 'test' else (img_fp_list, None)


def read_img(path: Union[str, Path]) -> torch.Tensor:
    """

    Args:
        path (str): image file path

    Returns: RGB torch.Tensor, with shape [Channel, Height, Width]
    """
    img = np.asarray(Image.open(path).convert('RGB'))
    return torch.tensor(img.transpose((2, 0, 1)))


def save_img(img: Union[Tensor, np.ndarray], path):
    if not isinstance(img, Tensor):
        img = torch.from_numpy(img)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    # img *= 255
    # img = img.to(dtype=torch.uint8)
    save_image(img, path)

    # Image.fromarray(img).save(path)
