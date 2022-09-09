# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).

import os
import logging
import glob
from multiprocessing import Process
import subprocess
from pprint import pformat

import click

from pix2text import set_logger, Pix2Text

_CONTEXT_SETTINGS = {"help_option_names": ['-h', '--help']}
logger = set_logger(log_level=logging.INFO)


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli():
    pass


@cli.command('predict')
@click.option(
    "-d",
    "--device",
    help="使用cpu还是 `gpu` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为 `cpu`",
    type=str,
    default='cpu',
)
@click.option("-i", "--img-file-or-dir", required=True, help="输入图片的文件路径或者指定的文件夹")
def predict(
    device, img_file_or_dir,
):
    """模型预测"""
    p2t = Pix2Text(device=device)

    fp_list = []
    if os.path.isfile(img_file_or_dir):
        fp_list.append(img_file_or_dir)
    elif os.path.isdir(img_file_or_dir):
        fn_list = glob.glob1(img_file_or_dir, '*g')
        fp_list = [os.path.join(img_file_or_dir, fn) for fn in fn_list]

    for fp in fp_list:
        out = p2t.recognize(fp)
        logger.info(f'In image {fp}, Out Text {pformat(out)}')


@cli.command('serve')
@click.option(
    '-H', '--host', type=str, default='0.0.0.0', help='server host. Default: "0.0.0.0"',
)
@click.option(
    '-p', '--port', type=int, default=8503, help='server port. Default: 8503',
)
@click.option(
    '--reload',
    is_flag=True,
    help='whether to reload the server when the codes have been changed',
)
def serve(host, port, reload):
    """开启HTTP服务。"""

    path = os.path.realpath(os.path.dirname(__file__))
    api = Process(
        target=start_server,
        kwargs={'path': path, 'host': host, 'port': port, 'reload': reload},
    )
    api.start()
    api.join()


def start_server(path, host, port, reload):
    cmd = ['uvicorn', 'serve:app', '--host', host, '--port', str(port)]
    if reload:
        cmd.append('--reload')
    subprocess.call(cmd, cwd=path)


if __name__ == "__main__":
    cli()
