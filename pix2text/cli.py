# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).

import os
import logging
import glob
import json
from multiprocessing import Process
import subprocess
from pprint import pformat

import click

from pix2text import set_logger, Pix2Text, merge_line_texts

_CONTEXT_SETTINGS = {"help_option_names": ['-h', '--help']}
logger = set_logger(log_level=logging.INFO)


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli():
    pass


@cli.command('predict')
@click.option(
    "-l",
    "--languages",
    type=str,
    default='en,ch_sim',
    help="Language Codes for Text-OCR to recognize, separated by commas",
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
    "--formula-ocr-config",
    type=str,
    default=None,
    help="Configuration information for the Latex-OCR mathematical formula recognition model. Default: `None`, meaning using the default configuration",
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
    "--image-type",
    type=click.Choice(['mixed', 'formula', 'text']),
    default='mixed',
    help="Which image type to process, either 'mixed', 'formula' or 'text'",
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
    analyzer_name,
    analyzer_type,
    analyzer_model_fp,
    formula_ocr_config,
    languages,
    text_ocr_config,
    device,
    image_type,
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

    formula_ocr_config = json.loads(formula_ocr_config) if formula_ocr_config else {}
    languages = [lang.strip() for lang in languages.split(',') if lang.strip()]
    text_ocr_config = json.loads(text_ocr_config) if text_ocr_config else {}
    p2t = Pix2Text(
        languages=languages,
        analyzer_config=analyzer_config,
        text_config=text_ocr_config,
        formula_config=formula_ocr_config,
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

    proc_func = {
        'mixed': p2t.recognize,
        'formula': p2t.recognize_formula,
        'text': p2t.recognize_text,
    }
    rec_kwargs = json.loads(rec_kwargs) if rec_kwargs else {}
    for idx, fp in enumerate(fp_list):
        analysis_res = save_analysis_res[idx] if save_analysis_res is not None else None
        out = proc_func[image_type](
            fp,
            resized_shape=resized_shape,
            save_analysis_res=analysis_res,
            **rec_kwargs,
        )
        if image_type == 'mixed':
            res = merge_line_texts(out, auto_line_break=auto_line_break)
        else:
            res = out
        logger.info(f'In image: {fp}\nOuts: \n\t{pformat(out)}\nOnly texts: \n{res}')


@cli.command('serve')
@click.option(
    "-l",
    "--languages",
    type=str,
    default='en,ch_sim',
    help="Language Codes for Text-OCR to recognize, separated by commas",
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
    "--formula-ocr-config",
    type=str,
    default=None,
    help="Configuration information for the Latex-OCR mathematical formula recognition model. Default: `None`, meaning using the default configuration",
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
@click.option(
    "--log-level",
    default='INFO',
    help="Log Level, such as `INFO`, `DEBUG`",
    show_default=True,
)
def serve(
    analyzer_name,
    analyzer_type,
    analyzer_model_fp,
    formula_ocr_config,
    languages,
    text_ocr_config,
    device,
    host,
    port,
    reload,
    log_level,
):
    """Start the HTTP service."""
    from pix2text.serve import start_server

    logger = set_logger(log_level=log_level)

    analyzer_config = dict(model_name=analyzer_name, model_type=analyzer_type)
    if analyzer_model_fp is not None:
        analyzer_config['model_fp'] = analyzer_model_fp

    formula_ocr_config = json.loads(formula_ocr_config) if formula_ocr_config else {}
    languages = [lang.strip() for lang in languages.split(',') if lang.strip()]
    text_ocr_config = json.loads(text_ocr_config) if text_ocr_config else {}
    p2t_config = dict(
        languages=languages,
        analyzer_config=analyzer_config,
        text_config=text_ocr_config,
        formula_config=formula_ocr_config,
        device=device,
    )
    api = Process(
        target=start_server,
        kwargs={'p2t_config': p2t_config, 'host': host, 'port': port, 'reload': reload},
    )
    api.start()
    api.join()


if __name__ == "__main__":
    cli()
