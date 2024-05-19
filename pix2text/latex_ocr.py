# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).
import os
import shutil
import string
from typing import Optional, Union, List, Dict, Any
import logging
from pathlib import Path
import re

import torch
import tqdm
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from PIL import Image
from cnocr.utils import get_default_ort_providers
from transformers.generation import (
    GenerateEncoderDecoderOutput,
    GenerateBeamEncoderDecoderOutput,
)

from .consts import MODEL_VERSION, AVAILABLE_MODELS
from .utils import data_dir, select_device, prepare_imgs

logger = logging.getLogger(__name__)


class LatexOCR(object):
    """Get a prediction of a math formula image in the easiest way"""

    def __init__(
        self,
        *,
        model_name: str = 'mfr',
        model_backend: str = 'onnx',
        device: str = None,
        context: str = None,  # deprecated, use `device` instead
        model_dir: Optional[Union[str, Path]] = None,
        root: Union[str, Path] = data_dir(),
        more_processor_configs: Optional[Dict[str, Any]] = None,
        more_model_configs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize a LatexOCR model.

        Args:
            model_name (str, optional): The name of the model. Defaults to 'mfr'.
            model_backend (str, optional): The model backend, either 'onnx' or 'pytorch'. Defaults to 'onnx'.
            device (str, optional): What device to use for computation, supports `['cpu', 'cuda', 'gpu']`; defaults to None, which selects the device automatically.
            context (str, optional): Deprecated, use `device` instead. What device to use for computation, supports `['cpu', 'cuda', 'gpu']`; defaults to None, which selects the device automatically.
            model_dir (Optional[Union[str, Path]], optional): The model file directory. Defaults to None.
            root (Union[str, Path], optional): The model root directory. Defaults to data_dir().
            more_processor_configs (Optional[Dict[str, Any]], optional): Additional processor configurations. Defaults to None.
            more_model_configs (Optional[Dict[str, Any]], optional): Additional model configurations. Defaults to None.

                - provider (`str`, defaults to `None`, which means to select one provider automatically):
                    ONNX Runtime provider to use for loading the model. See https://onnxruntime.ai/docs/execution-providers/ for
                    possible providers.
                - session_options (`Optional[onnxruntime.SessionOptions]`, defaults to `None`),:
                    ONNX Runtime session options to use for loading the model.
                - provider_options (`Optional[Dict[str, Any]]`, defaults to `None`):
                    Provider option dictionaries corresponding to the provider used. See available options
                    for each provider: https://onnxruntime.ai/docs/api/c/group___global.html .
                - ...: see more information here: optimum.onnxruntime.modeling_ort.ORTModel.from_pretrained()
            **kwargs: Additional arguments, currently not used.
        """

        if context is not None:
            logger.warning(f'`context` is deprecated, please use `device` instead')
        if device is None and context is not None:
            device = context
        self.device = select_device(device)

        model_info = AVAILABLE_MODELS.get_info(model_name, model_backend)

        if model_dir is None:
            model_dir = self._prepare_model_files(root, model_backend, model_info)
        logger.info(f'Use model dir for LatexOCR: {model_dir}')

        more_model_configs = more_model_configs or {}
        if model_backend == 'onnx' and 'provider' not in more_model_configs:
            available_providers = get_default_ort_providers()
            if not available_providers:
                raise RuntimeError(
                    'No available providers for ONNX Runtime, please install onnxruntime-gpu or onnxruntime.'
                )
            more_model_configs['provider'] = available_providers[0]
        self.model, self.processor = self._init_model(
            model_backend,
            model_dir,
            more_processor_config=more_processor_configs,
            more_model_config=more_model_configs,
        )
        logger.info(
            f'Loaded Pix2Text MFR model {model_name} to: backend-{model_backend}, device-{self.device}'
        )

    def _prepare_model_files(self, root, model_backend, model_info):
        model_root_dir = Path(root) / MODEL_VERSION
        model_dir = model_root_dir / model_info['local_model_id']
        if model_dir.is_dir():
            return str(model_dir)
        assert 'hf_model_id' in model_info
        model_dir.mkdir(parents=True)
        download_cmd = f'huggingface-cli download --repo-type model --resume-download --local-dir-use-symlinks False {model_info["hf_model_id"]} --local-dir {model_dir}'
        os.system(download_cmd)
        # 如果当前目录下无文件，就从huggingface上下载
        if not list(model_dir.glob('**/[!.]*')):
            if model_dir.exists():
                shutil.rmtree(str(model_dir))
            os.system('HF_ENDPOINT=https://hf-mirror.com ' + download_cmd)
        return model_dir

    def _init_model(
        self,
        model_backend,
        model_dir,
        more_processor_config=None,
        more_model_config=None,
    ):
        more_processor_config = more_processor_config or {}
        more_model_config = more_model_config or {}
        processor = TrOCRProcessor.from_pretrained(model_dir, **more_processor_config)
        if model_backend == 'pytorch':
            model = VisionEncoderDecoderModel.from_pretrained(
                model_dir, **more_model_config
            )
            model.to(self.device)
            model.eval()
        else:
            if 'use_cache' not in more_model_config:
                more_model_config['use_cache'] = False
            if (
                'provider' in more_model_config
                and more_model_config['provider'] == 'CUDAExecutionProvider'
            ):
                more_model_config["use_io_binding"] = more_model_config['use_cache']
            model = ORTModelForVision2Seq.from_pretrained(
                model_dir, **more_model_config
            )
            model.to(self.device)
        return model, processor

    def __call__(self, *args, **kwargs) -> Union[str, List[str]]:
        return self.recognize(*args, **kwargs)

    def recognize(
        self,
        imgs: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]],
        batch_size: int = 1,
        use_post_process: bool = True,
        rec_config: Optional[dict] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Recognize Math Formula images to LaTeX Expressions
        Args:
            imgs (Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]): The image or list of images
            batch_size (int): The batch size
            use_post_process (bool): Whether to use post process. Defaults to True
            rec_config (Optional[dict]): The generation config
            **kwargs (): Other arguments. Not used for now

        Returns: The LaTeX Result or list of LaTeX Results; each result is a dict with `text` and `score` fields.

        """
        is_single_image = False
        if isinstance(imgs, (str, Path, Image.Image)):
            imgs = [imgs]
            is_single_image = True

        input_imgs = prepare_imgs(imgs)

        # inference batch by batch
        results = []
        for i in tqdm.tqdm(range(0, len(input_imgs), batch_size)):
            part_imgs = input_imgs[i : i + batch_size]
            results.extend(self._one_batch(part_imgs, rec_config, **kwargs))

        if use_post_process:
            for info in results:
                info['text'] = self.post_process(info['text'])

        if is_single_image:
            return results[0]
        return results

    def _one_batch(self, img_list, rec_config, **kwargs):
        rec_config = rec_config or {}
        pixel_values = self.processor(images=img_list, return_tensors="pt").pixel_values
        outs = self.model.generate(
            pixel_values.to(self.device),
            return_dict_in_generate=True,
            output_scores=True,
            **rec_config,
        )
        mean_probs = self._cal_scores(outs)

        generated_text = self.processor.batch_decode(
            outs.sequences, skip_special_tokens=True
        )
        assert len(img_list) == len(generated_text) == len(mean_probs)

        final_out = []
        for text, prob in zip(generated_text, mean_probs):
            final_out.append({'text': text, 'score': prob})
        return final_out

    def _cal_scores(self, outs):
        if isinstance(outs, GenerateBeamEncoderDecoderOutput):
            mean_probs = outs.sequences_scores.exp().tolist()
        elif isinstance(outs, GenerateEncoderDecoderOutput):
            logits = torch.stack(outs.scores, dim=1)
            scores = torch.softmax(logits, dim=-1).max(dim=2).values

            mean_probs = []
            for idx, example in enumerate(scores):
                cur_length = int(
                    (outs.sequences[idx] != self.processor.tokenizer.pad_token_id).sum()
                )
                assert cur_length > 1
                # 获得几何平均值。注意：example中的第一个元素对应sequence中的第二个元素
                mean_probs.append(
                    float((example[: cur_length - 1] + 1e-8).log().mean().exp())
                )
        else:
            raise Exception(f'unprocessed output type: {type(outs)}')

        return mean_probs

    def post_process(self, text):
        text = remove_redundant_script(text)
        text = remove_trailing_whitespace(text)
        text = replace_illegal_symbols(text)
        for _ in range(10):
            new_text = remove_empty_text(text)
            if new_text == text:
                break
            text = new_text

        text = fix_latex(text)
        text = remove_unnecessary_spaces(text)
        return text.strip()


def remove_redundant_script(text):
    # change '^ { abc }' to 'abc'
    pattern = r'^\^\s*{\s*(.*?)\s*}'
    result = re.sub(pattern, r'\1', text)
    # change '_ { abc }' to 'abc'
    pattern = r'^_\s*{\s*(.*?)\s*}'
    result = re.sub(pattern, r'\1', result)
    return result.strip()


def replace_illegal_symbols(text):
    illegal_to_legals = [
        (r'\\\.', r'\\ .'),  # \. -> \ .
        (r'\\=', r'\\ ='),  # \= -> \ =
        (r'\\-', r'\\ -'),  # \- -> \ -
        (r'\\~', r'\\ ~'),  # \~ -> \ ~
    ]
    for illegal, legal in illegal_to_legals:
        text = re.sub(illegal, legal, text)
    return text


def remove_empty_text(latex_expression):
    # change 'abc ^' to 'abc'
    patterns = [
        r'\\hat\s*{\s*}',  # 匹配 \hat{}
        r'\^\s*{\s*}',  # 匹配 ^{}
        r'_\s*{\s*}',  # 匹配 _{}
        r'\\text\s*{\s*}',  # 匹配 \text{}
        r'\\tilde\s*{\s*}',  # 匹配 \tilde{}
        r'\\bar\s*{\s*}',  # 匹配 \bar{}
        r'\\vec\s*{\s*}',  # 匹配 \vec{}
        r'\\acute\s*{\s*}',  # 匹配 \acute{}
        r'\\grave\s*{\s*}',  # 匹配 \grave{}
        r'\\breve\s*{\s*}',  # 匹配 \breve{}
        r'\\overline\s*{\s*}',  # 匹配 \overline{}
        r'\\dot\s*{\s*}',  # 匹配 \dot{}
        r'\\ddot\s*{\s*}',  # 匹配 \ddot{}
        r'\\widehat\s*{\s*}',  # 匹配 \widehat{}
        r'\\widetilde\s*{\s*}',  # 匹配 \widetilde{}
    ]

    # 使用 sub 函数进行替换
    for pattern in patterns:
        latex_expression = re.sub(pattern, '', latex_expression)
    return latex_expression.strip()


latex_whitespace_symbols = [
    r'\\ +',  # 空格
    r'\\quad\s*',  # 1em 宽度的空白
    r'\\qquad\s*',  # 2em 宽度的空白
    r'\\,\s*',  # 窄空格
    r'\\:\s*',  # 中等空格
    r'\\;\s*',  # 大空格
    r'\\enspace\s*',  # 1em 空白
    r'\\thinspace\s*',  # 1/6em 空白
    r'\\!\s*',  # 感叹号
]


def remove_trailing_whitespace(latex_str):
    # 定义匹配末尾空白符号的正则表达式模式
    trailing_whitespace_pattern = r'(?:' + '|'.join(latex_whitespace_symbols) + r')+$'

    # 使用 sub 函数进行替换
    return re.sub(trailing_whitespace_pattern, '', latex_str).strip()


def remove_unnecessary_spaces(latex_str):
    # Preserve space between a command and a following uppercase letter
    latex_str = re.sub(r'\\([a-zA-Z]+) (?=[a-zA-Z])', r'\\\1 ', latex_str)

    # Remove spaces after commands not followed by an uppercase letter, carefully not affecting commands that require space
    latex_str = re.sub(r'\\([a-zA-Z]+)\s+(?![a-zA-Z])', r'\\\1', latex_str)

    # Remove spaces around curly braces, preserving internal spaces
    latex_str = re.sub(r'(\{)\s+', r'\1', latex_str)
    latex_str = re.sub(r'\s+(\})', r'\1', latex_str)

    # Specifically target and remove spaces around mathematical operators, including +, -, =, and similar operators
    latex_str = re.sub(r'(?<=[^\\])\s*([+\-=])\s*', r'\1', latex_str)

    # Remove spaces around "^" and "_" for subscripts and superscripts
    latex_str = re.sub(r'\s*(\^|\_)\s*', r'\1', latex_str)

    return latex_str


def find_all_left_or_right(latex, left_or_right='left'):
    left_bracket_infos = []
    prefix_len = len(left_or_right) + 1
    # 匹配出latex中所有的 '\left' 后面跟着的第一个非空格字符，定位它们所在的位置
    for m in re.finditer(rf'\\{left_or_right}\s*\S', latex):
        start, end = m.span()
        if latex[end - 1] in string.ascii_letters:
            continue
        # 如果最后一个字符为 "\"，则往前继续匹配，直到匹配到一个非字母的字符
        # 如 "\left \big("
        while latex[end - 1] in ('\\', ' '):
            end += 1
            while end < len(latex) and latex[end].isalpha():
                end += 1
        ori_str = latex[start + prefix_len : end].strip()
        # FIXME: ori_str中可能出现多个 '\left'，此时需要分隔开

        left_bracket_infos.append({'str': ori_str, 'start': start, 'end': end})
        left_bracket_infos.sort(key=lambda x: x['start'])
    return left_bracket_infos


def match_left_right(left_str, right_str):
    """匹配左右括号，如匹配 `\left(` 和 `\right)`。"""
    left_str = left_str.strip().replace(' ', '')[len('left') + 1 :]
    right_str = right_str.strip().replace(' ', '')[len('right') + 1 :]
    # 去掉开头的相同部分
    while left_str and right_str and left_str[0] == right_str[0]:
        left_str = left_str[1:]
        right_str = right_str[1:]

    match_pairs = [
        ('', ''),
        ('(', ')'),
        ('\{', '.'),  # 大括号那种
        ('⟮', '⟯'),
        ('[', ']'),
        ('⟨', '⟩'),
        ('{', '}'),
        ('⌈', '⌉'),
        ('┌', '┐'),
        ('⌊', '⌋'),
        ('└', '┘'),
        ('⎰', '⎱'),
        ('lt', 'gt'),
        ('lang', 'rang'),
        (r'langle', r'rangle'),
        (r'lbrace', r'rbrace'),
        ('lBrace', 'rBrace'),
        (r'lbracket', r'rbracket'),
        (r'lceil', r'rceil'),
        ('lcorner', 'rcorner'),
        (r'lfloor', r'rfloor'),
        (r'lgroup', r'rgroup'),
        (r'lmoustache', r'rmoustache'),
        (r'lparen', r'rparen'),
        (r'lvert', r'rvert'),
        (r'lVert', r'rVert'),
    ]
    return (left_str, right_str) in match_pairs


def fix_latex(latex: str) -> str:
    """对识别结果做进一步处理和修正。"""
    # # 把latex中的中文括号全部替换成英文括号
    # latex = latex.replace('（', '(').replace('）', ')')
    # # 把latex中的中文逗号全部替换成英文逗号
    # latex = latex.replace('，', ',')

    left_bracket_infos = find_all_left_or_right(latex, left_or_right='left')
    right_bracket_infos = find_all_left_or_right(latex, left_or_right='right')
    # left 和 right 找配对，left找位置比它靠前且最靠近他的right配对
    for left_bracket_info in left_bracket_infos:
        for right_bracket_info in right_bracket_infos:
            if (
                not right_bracket_info.get('matched', False)
                and right_bracket_info['start'] > left_bracket_info['start']
                and match_left_right(
                    right_bracket_info['str'], left_bracket_info['str']
                )
            ):
                left_bracket_info['matched'] = True
                right_bracket_info['matched'] = True
                break

    for left_bracket_info in left_bracket_infos:
        # 把没有匹配的 '\left'替换为等长度的空格
        left_len = len('left') + 1
        if not left_bracket_info.get('matched', False):
            start_idx = left_bracket_info['start']
            end_idx = start_idx + left_len
            latex = (
                latex[: left_bracket_info['start']]
                + ' ' * (end_idx - start_idx)
                + latex[end_idx:]
            )
    for right_bracket_info in right_bracket_infos:
        # 把没有匹配的 '\right'替换为等长度的空格
        right_len = len('right') + 1
        if not right_bracket_info.get('matched', False):
            start_idx = right_bracket_info['start']
            end_idx = start_idx + right_len
            latex = (
                latex[: right_bracket_info['start']]
                + ' ' * (end_idx - start_idx)
                + latex[end_idx:]
            )

    # 把 latex 中的连续空格替换为一个空格
    latex = re.sub(r'\s+', ' ', latex)
    return latex.strip()
