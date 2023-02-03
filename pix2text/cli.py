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
    "--use-analyzer/--no-use-analyzer",
    default=True,
    help="是否使用 MFD 或者版面分析 Analyzer",
    show_default=True,
)
@click.option(
    "-a",
    "--analyzer-name",
    type=click.Choice(['mfd', 'layout']),
    default='mfd',
    help="使用哪个Analyzer，MFD还是版面分析",
    show_default=True,
)
@click.option(
    "-d",
    "--device",
    help="使用 `cpu` 还是 `gpu` 运行代码，也可指定为特定gpu，如`cuda:0`",
    type=str,
    default='cpu',
    show_default=True,
)
@click.option(
    "--resized-shape", help="把图片宽度resize到此大小再进行处理", type=int, default=700,
    show_default=True,
)
@click.option("-i", "--img-file-or-dir", required=True, help="输入图片的文件路径或者指定的文件夹")
@click.option(
    "--save-analysis-res",
    default=None,
    help="把解析结果存储到此文件或目录中"
         "（如果'--img-file-or-dir'为文件/文件夹，则'--save-analysis-res'也应该是文件/文件夹）。"
         "取值为 `None` 表示不存储",
    show_default=True,
)
@click.option(
    "-l",
    "--log-level",
    default='INFO',
    help="Log Level, such as `INFO`, `DEBUG`",
    show_default=True,
)
def predict(
    use_analyzer,
    analyzer_name,
    device,
    resized_shape,
    img_file_or_dir,
    save_analysis_res,
    log_level,
):
    """模型预测"""
    logger = set_logger(log_level=log_level)

    p2t = Pix2Text(analyzer_config=dict(model_name=analyzer_name), device=device)

    fp_list = []
    if os.path.isfile(img_file_or_dir):
        fp_list.append(img_file_or_dir)
        if save_analysis_res:
            save_analysis_res = [save_analysis_res]
    elif os.path.isdir(img_file_or_dir):
        fn_list = glob.glob1(img_file_or_dir, '*g')
        fp_list = [os.path.join(img_file_or_dir, fn) for fn in fn_list]
        if save_analysis_res:
            os.makedirs(save_analysis_res, exist_ok=True)
            save_analysis_res = [os.path.join(save_analysis_res, 'analysis-' + fn) for fn in fn_list]

    for idx, fp in enumerate(fp_list):
        analysis_res = save_analysis_res[idx] if save_analysis_res is not None else None
        out = p2t.recognize(
            fp,
            use_analyzer=use_analyzer,
            resized_shape=resized_shape,
            save_analysis_res=analysis_res,
        )
        res = '\n'.join([o['text'] for o in out])
        logger.info(f'In image: {fp}\nOuts: \n\t{pformat(out)}\nOnly texts: \n\t{res}')


@cli.command('serve')
@click.option(
    '-H', '--host', type=str, default='0.0.0.0', help='server host',
    show_default=True,
)
@click.option(
    '-p', '--port', type=int, default=8503, help='server port',
    show_default=True,
)
@click.option(
    '--reload',
    is_flag=True,
    help='whether to reload the server when the codes have been changed',
    show_default=True,
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
