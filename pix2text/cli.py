# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).

import os
import logging
import glob
import json
from multiprocessing import Process
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
    "-l",
    "--languages",
    type=str,
    default='en,ch_sim',
    help="Language Codes for Text-OCR to recognize, separated by commas",
    show_default=True,
)
@click.option(
    "--layout-config",
    type=str,
    default=None,
    help="Configuration information for the layout parser model, in JSON string format. Default: `None`, meaning using the default configuration",
    show_default=True,
)
@click.option(
    "--mfd-config",
    type=str,
    default=None,
    help="Configuration information for the MFD model, in JSON string format. Default: `None`, meaning using the default configuration",
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
    "--enable-formula/--disable-formula",
    default=True,
    help="Whether to enable formula recognition",
    show_default=True,
)
@click.option(
    "--enable-table/--disable-table",
    default=True,
    help="Whether to enable table recognition",
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
    "--file-type",
    type=click.Choice(['pdf', 'page', 'text_formula', 'formula', 'text']),
    default='text_formula',
    help="Which file type to process, 'pdf', 'page', 'text_formula', 'formula', or 'text'",
    show_default=True,
)
@click.option(
    "--resized-shape",
    help="Resize the image width to this size before processing",
    type=int,
    default=768,
    show_default=True,
)
@click.option(
    "-i",
    "--img-file-or-dir",
    required=True,
    help="File path of the input image/pdf or the specified directory",
)
@click.option(
    "--save-debug-res",
    default=None,
    help="If `save_debug_res` is set, the directory to save the debug results; default value is `None`, which means not to save",
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
    "--return-text/--no-return-text",
    default=True,
    help="Whether to return only the text result",
    show_default=True,
)
@click.option(
    "--auto-line-break/--no-auto-line-break",
    default=True,
    help="Whether to automatically determine to merge adjacent line results into a single line result",
    show_default=True,
)
@click.option(
    "-o",
    "--output-dir",
    default='output-md',
    help="Output directory for the recognized text results. Only effective when `file-type` is `pdf` or `page`",
    show_default=True,
)
@click.option(
    "--log-level",
    default='INFO',
    help="Log Level, such as `INFO`, `DEBUG`",
    show_default=True,
)
def predict(
    languages,
    layout_config,
    mfd_config,
    formula_ocr_config,
    text_ocr_config,
    enable_formula,
    enable_table,
    device,
    file_type,
    resized_shape,
    img_file_or_dir,
    save_debug_res,
    rec_kwargs,
    return_text,
    auto_line_break,
    output_dir,
    log_level,
):
    """Use Pix2Text (P2T) to predict the text information in an image or PDF."""
    logger = set_logger(log_level=log_level)

    analyzer_config = json.loads(mfd_config) if mfd_config else {}
    formula_ocr_config = json.loads(formula_ocr_config) if formula_ocr_config else {}
    languages = [lang.strip() for lang in languages.split(',') if lang.strip()]
    text_ocr_config = json.loads(text_ocr_config) if text_ocr_config else {}

    layout_config = json.loads(layout_config) if layout_config else {}
    text_formula_config = {
        'languages': languages,  # 'en,ch_sim
        'mfd': analyzer_config,
        'formula': formula_ocr_config,
        'text': text_ocr_config,
    }
    total_config = {
        'layout': layout_config,
        'text_formula': text_formula_config,
    }
    p2t = Pix2Text.from_config(
        total_configs=total_config,
        enable_formula=enable_formula,
        enable_table=enable_table,
        device=device,
    )

    fp_list = []
    if os.path.isfile(img_file_or_dir):
        fp_list.append(img_file_or_dir)
        if save_debug_res:
            save_debug_res = [save_debug_res]
    elif os.path.isdir(img_file_or_dir):
        fn_list = glob.glob1(img_file_or_dir, '*g')
        fp_list = [os.path.join(img_file_or_dir, fn) for fn in fn_list]
        if save_debug_res:
            os.makedirs(save_debug_res, exist_ok=True)
            save_debug_res = [
                os.path.join(save_debug_res, 'output-debugs-' + fn) for fn in fn_list
            ]
    else:
        raise ValueError(f'{img_file_or_dir} is not a valid file or directory')

    rec_kwargs = json.loads(rec_kwargs) if rec_kwargs else {}
    rec_kwargs['resized_shape'] = resized_shape
    rec_kwargs['return_text'] = return_text
    rec_kwargs['auto_line_break'] = auto_line_break

    for idx, fp in enumerate(fp_list):
        if file_type in ('pdf', 'page'):
            rec_kwargs['save_debug_res'] = (
                save_debug_res[idx] if save_debug_res is not None else None
            )
        else:
            rec_kwargs['save_analysis_res'] = (
                save_debug_res[idx] if save_debug_res is not None else None
            )
        out = p2t.recognize(fp, file_type=file_type, **rec_kwargs)
        if file_type in ('pdf', 'page'):
            out = out.to_markdown(output_dir)
        logger.info(
            f'In image: {fp}\nOuts: \n{out if isinstance(out, str) else pformat(out)}\n'
        )


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
    "--layout-config",
    type=str,
    default=None,
    help="Configuration information for the layout parser model, in JSON string format. Default: `None`, meaning using the default configuration",
    show_default=True,
)
@click.option(
    "--mfd-config",
    type=str,
    default=None,
    help="Configuration information for the MFD model, in JSON string format. Default: `None`, meaning using the default configuration",
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
    "--enable-formula/--disable-formula",
    default=True,
    help="Whether to enable formula recognition",
    show_default=True,
)
@click.option(
    "--enable-table/--disable-table",
    default=True,
    help="Whether to enable table recognition",
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
    "-o",
    "--output-md-root-dir",
    default='output-md-root',
    help="Markdown output root directory for the recognized text results. Only effective when `file-type` is `pdf` or `page`",
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
    languages,
    layout_config,
    mfd_config,
    formula_ocr_config,
    text_ocr_config,
    enable_formula,
    enable_table,
    device,
    output_md_root_dir,
    host,
    port,
    reload,
    log_level,
):
    """Start the HTTP service."""
    from pix2text.serve import start_server

    logger = set_logger(log_level=log_level)

    analyzer_config = json.loads(mfd_config) if mfd_config else {}
    formula_ocr_config = json.loads(formula_ocr_config) if formula_ocr_config else {}
    languages = [lang.strip() for lang in languages.split(',') if lang.strip()]
    text_ocr_config = json.loads(text_ocr_config) if text_ocr_config else {}

    layout_config = json.loads(layout_config) if layout_config else {}
    text_formula_config = {
        'languages': languages,  # 'en,ch_sim
        'mfd': analyzer_config,
        'formula': formula_ocr_config,
        'text': text_ocr_config,
    }
    total_config = {
        'layout': layout_config,
        'text_formula': text_formula_config,
    }
    p2t_config = dict(
        total_configs=total_config,
        enable_formula=enable_formula,
        enable_table=enable_table,
        device=device,
    )
    api = Process(
        target=start_server,
        kwargs={
            'p2t_config': p2t_config,
            'output_md_root_dir': output_md_root_dir,
            'host': host,
            'port': port,
            'reload': reload,
        },
    )
    api.start()
    api.join()


if __name__ == "__main__":
    cli()
