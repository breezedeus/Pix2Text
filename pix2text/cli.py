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

    mfd_config = json.loads(mfd_config) if mfd_config else {}
    formula_ocr_config = json.loads(formula_ocr_config) if formula_ocr_config else {}
    languages = [lang.strip() for lang in languages.split(',') if lang.strip()]
    text_ocr_config = json.loads(text_ocr_config) if text_ocr_config else {}

    layout_config = json.loads(layout_config) if layout_config else {}
    text_formula_config = {
        'languages': languages,  # 'en,ch_sim
        'mfd': mfd_config,
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


@cli.command('evaluate')
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
    "--input-json",
    required=True,
    help="JSON file containing evaluation data with image paths and ground truth",
)
@click.option(
    "--gt-key",
    default="model_result",
    help="Key name for ground truth text in the JSON data",
    show_default=True,
)
@click.option(
    "--prefix-img-dir",
    default="data",
    help="Root directory for image files, will be prepended to img_path in JSON",
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
    default=True,
    help="Whether to automatically determine to merge adjacent line results into a single line result",
    show_default=True,
)
@click.option(
    "-o",
    "--output-json",
    default='evaluation_results.json',
    help="Output JSON file for evaluation results",
    show_default=True,
)
@click.option(
    "--output-excel",
    default=None,
    help="Output Excel file with embedded images (optional)",
    show_default=True,
)
@click.option(
    "--output-html",
    default=None,
    help="Output HTML report with embedded images (optional)",
    show_default=True,
)
@click.option(
    "--max-img-width",
    default=400,
    help="Maximum width for embedded images in pixels",
    show_default=True,
)
@click.option(
    "--max-img-height",
    default=300,
    help="Maximum height for embedded images in pixels",
    show_default=True,
)
@click.option(
    "--max-samples",
    default=-1,
    help="Maximum number of samples to process (-1 for all samples)",
    show_default=True,
)
@click.option(
    "--log-level",
    default='INFO',
    help="Log Level, such as `INFO`, `DEBUG`",
    show_default=True,
)
def evaluate(
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
    input_json,
    gt_key,
    prefix_img_dir,
    rec_kwargs,
    auto_line_break,
    output_json,
    output_excel,
    output_html,
    max_img_width,
    max_img_height,
    max_samples,
    log_level,
):
    """Evaluate Pix2Text (P2T) performance using a JSON file with image paths and ground truth."""
    from pix2text.utils import (
        calculate_cer_batch, 
        calculate_cer,
        save_evaluation_results_to_excel_with_images,
        create_html_report_with_images
    )
    
    logger = set_logger(log_level=log_level)

    # Load evaluation data
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load evaluation data from {input_json}: {e}")
        return

    if not isinstance(eval_data, list):
        logger.error("Evaluation data must be a list of dictionaries")
        return

    # Validate data format
    for i, item in enumerate(eval_data):
        if not isinstance(item, dict):
            logger.error(f"Item {i} is not a dictionary")
            return
        if 'img_path' not in item or gt_key not in item:
            logger.error(f"Item {i} missing required keys 'img_path' or '{gt_key}'")
            return

    # Initialize Pix2Text
    mfd_config = json.loads(mfd_config) if mfd_config else {}
    formula_ocr_config = json.loads(formula_ocr_config) if formula_ocr_config else {}
    languages = [lang.strip() for lang in languages.split(',') if lang.strip()]
    text_ocr_config = json.loads(text_ocr_config) if text_ocr_config else {}

    layout_config = json.loads(layout_config) if layout_config else {}
    text_formula_config = {
        'languages': languages,
        'mfd': mfd_config,
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

    # Prepare recognition kwargs
    rec_kwargs = json.loads(rec_kwargs) if rec_kwargs else {}
    rec_kwargs['resized_shape'] = resized_shape
    rec_kwargs['return_text'] = True
    rec_kwargs['auto_line_break'] = auto_line_break

    def filter_and_clean_gt(gt):
        # 只针对部分的图片进行识别
        # 去掉收尾的'"'
        if not gt:
            return False, gt
        if gt.startswith(r'$$') and gt.endswith(r'$$'):
            gt = gt[2:-2]
            if '$$' not in gt:
                return True, gt.strip()
        return False, gt

    # Process each image and collect results
    predictions = []
    ground_truths = []
    results = []

    # Apply max_samples limit
    if max_samples > 0:
        import random
        random.seed(42)
        random.shuffle(eval_data)
    
    logger.info(f"Limited to {max_samples} samples for evaluation")
    logger.info(f"Starting evaluation on {len(eval_data)} images...")

    for i, item in enumerate(eval_data):
        if len(results) >= max_samples:
            break
        img_path = item['new_img_path']
        ground_truth = item[gt_key]
        
        # Handle ground truth that might be a JSON string
        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except json.JSONDecodeError:
                # If it's not valid JSON, use as is
                pass
        
        # Apply formula filtering if needed
        is_formula, ground_truth = filter_and_clean_gt(ground_truth)
        if not is_formula:
            continue

        # Prepend prefix_img_dir to img_path if it's not an absolute path
        if not os.path.isabs(img_path):
            img_path = os.path.join(prefix_img_dir, img_path)
        
        logger.info(f"Processing image {i+1}/{len(eval_data)}: {img_path}")
        
        try:
            # Check if image file exists
            if not os.path.exists(img_path):
                logger.warning(f"Image file not found: {img_path}")
                continue
                
            # Recognize text
            prediction = p2t.recognize(img_path, file_type=file_type, **rec_kwargs)
            
            # Convert to string if needed
            if not isinstance(prediction, str):
                if hasattr(prediction, 'to_markdown'):
                    prediction = prediction.to_markdown()
                else:
                    prediction = str(prediction)
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
            
            # Calculate individual CER
            cer = calculate_cer(prediction, ground_truth)
            
            result = {
                'img_path': img_path,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'cer': cer
            }
            results.append(result)
            
            logger.info(f"Image {img_path} CER: {cer:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            continue

    # resort results by cer
    # results.sort(key=lambda x: x['cer'], reverse=True)

    # Calculate overall CER
    if predictions and ground_truths:
        cer_stats = calculate_cer_batch(predictions, ground_truths)
        
        # Prepare final results
        evaluation_results = {
            'summary': {
                'total_samples': len(results),
                'average_cer': cer_stats['average_cer'],
                'individual_cers': cer_stats['individual_cers']
            },
            'detailed_results': results
        }
        
        # Save results
        try:
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Evaluation results saved to: {output_json}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
        
        # Print summary
        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total samples processed: {len(results)}")
        logger.info(f"Average CER: {cer_stats['average_cer']:.4f}")
        logger.info(f"Best CER: {min(cer_stats['individual_cers']):.4f}")
        logger.info(f"Worst CER: {max(cer_stats['individual_cers']):.4f}")
        logger.info("=" * 50)
        
    else:
        logger.error("No valid predictions generated")

    # Save results to Excel with embedded images (if requested)
    if output_excel and results:
        excel_success = save_evaluation_results_to_excel_with_images(
            results=results,
            output_file=output_excel,
            img_path_key='img_path',
            gt_key='ground_truth',
            pred_key='prediction',
            cer_key='cer',
            max_img_width=max_img_width,
            max_img_height=max_img_height
        )
        if excel_success:
            logger.info(f"Excel file with embedded images saved to: {output_excel}")
        else:
            logger.warning("Failed to save Excel file with embedded images")
    
    # Save results to HTML report with embedded images (if requested)
    if output_html and results:
        html_success = create_html_report_with_images(
            results=results,
            output_file=output_html,
            img_path_key='img_path',
            gt_key='ground_truth',
            pred_key='prediction',
            cer_key='cer',
            max_img_width=max_img_width,
            max_img_height=max_img_height
        )
        if html_success:
            logger.info(f"HTML report with embedded images saved to: {output_html}")
        else:
            logger.warning("Failed to save HTML report with embedded images")


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
