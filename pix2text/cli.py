# coding: utf-8
# Copyright (C) 2022-2023, [Breezedeus](https://www.breezedeus.com).

import os
import logging
import glob
import json
from multiprocessing import Process
import subprocess
from pprint import pformat

import click

from pix2text import set_logger, Pix2Text, merge_line_texts
from pix2text.consts import LATEX_CONFIG_FP

_CONTEXT_SETTINGS = {"help_option_names": ['-h', '--help']}
logger = set_logger(log_level=logging.INFO)


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli():
    pass


@cli.command('predict')
@click.option(
    "--use-analyzer/--no-use-analyzer",
    default=True,
    help="Whether to use MFD (Mathematical Formula Detection) or Layout Analysis",
    show_default=True,
)
@click.option(
    "-l",
    "--languages",
    type=str,
    default='en,ch_sim',
    help="Languages for Text-OCR to recognize, separated by commas",
    show_default=True,
)
@click.option(
    "-a",
    "--analyzer-name",
    type=click.Choice(['mfd', 'layout']),
    default='mfd',
    help="Which Analyzer to use, either MFD or Layout Analysis",
    show_default=True,
)
@click.option(
    "-t",
    "--analyzer-type",
    type=str,
    default='yolov7_tiny',
    help="Which model to use for the Analyzer, 'yolov7_tiny' or 'yolov7'",
    show_default=True,
)
@click.option(
    "--analyzer-model-fp",
    type=str,
    default=None,
    help="File path for the Analyzer detection model. Default: `None`, meaning using the default model",
    show_default=True,
)
@click.option(
    "--latex-ocr-model-fp",
    type=str,
    default=None,
    help="File path for the Latex-OCR mathematical formula recognition model. Default: `None`, meaning using the default model",
    show_default=True,
)
@click.option(
    "--text-ocr-config",
    type=str,
    default=None,
    help="Configuration information for Text-OCR recognition, in JSON string format. Default: `None`, meaning using the default configuration",
    show_default=True,
)
@click.option(
    "-d",
    "--device",
    help="Choose to run the code using `cpu`, `gpu`, or a specific GPU like `cuda:0`",
    type=str,
    default='cpu',
    show_default=True,
)
@click.option(
    "--resized-shape",
    help="Resize the image width to this size before processing",
    type=int,
    default=608,
    show_default=True,
)
@click.option(
    "-i",
    "--img-file-or-dir",
    required=True,
    help="File path of the input image or the specified directory",
)
@click.option(
    "--save-analysis-res",
    default=None,
    help="Save the analysis results to this file or directory"
    " (If '--img-file-or-dir' is a file/directory, then '--save-analysis-res' should also be a file/directory)."
    " Set to `None` for not saving",
    show_default=True,
)
@click.option(
    "--rec-kwargs",
    type=str,
    default=None,
    help="kwargs for calling .recognize(), in JSON string format",
    show_default=True,
)
@click.option(
    "--auto-line-break/--no-auto-line-break",
    default=False,
    help="Whether to automatically determine to merge adjacent line results into a single line result",
    show_default=True,
)
@click.option(
    "--log-level",
    default='INFO',
    help="Log Level, such as `INFO`, `DEBUG`",
    show_default=True,
)
def predict(
    use_analyzer,
    analyzer_name,
    analyzer_type,
    analyzer_model_fp,
    latex_ocr_model_fp,
    languages,
    text_ocr_config,
    device,
    resized_shape,
    img_file_or_dir,
    save_analysis_res,
    rec_kwargs,
    auto_line_break,
    log_level,
):
    """Use Pix2Text (P2T) to predict the text information in an image"""
    logger = set_logger(log_level=log_level)

    analyzer_config = dict(model_name=analyzer_name, model_type=analyzer_type)
    if analyzer_model_fp is not None:
        analyzer_config['model_fp'] = analyzer_model_fp

    formula_config = None
    if latex_ocr_model_fp is not None:
        formula_config = {'model_fp': latex_ocr_model_fp}
    languages = [lang.strip() for lang in languages.split(',') if lang.strip()]
    text_ocr_config = json.loads(text_ocr_config) if text_ocr_config else {}
    p2t = Pix2Text(
        languages=languages,
        analyzer_config=analyzer_config,
        text_config=text_ocr_config,
        formula_config=formula_config,
        device=device,
    )

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
            save_analysis_res = [
                os.path.join(save_analysis_res, 'analysis-' + fn) for fn in fn_list
            ]

    rec_kwargs = json.loads(rec_kwargs) if rec_kwargs else {}
    for idx, fp in enumerate(fp_list):
        analysis_res = save_analysis_res[idx] if save_analysis_res is not None else None
        out = p2t.recognize(
            fp,
            use_analyzer=use_analyzer,
            resized_shape=resized_shape,
            save_analysis_res=analysis_res,
            **rec_kwargs,
        )
        res = merge_line_texts(out, auto_line_break=auto_line_break)
        logger.info(f'In image: {fp}\nOuts: \n\t{pformat(out)}\nOnly texts: \n{res}')


@cli.command('serve')
@click.option(
    '-H', '--host', type=str, default='0.0.0.0', help='server host', show_default=True,
)
@click.option(
    '-p', '--port', type=int, default=8503, help='server port', show_default=True,
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
