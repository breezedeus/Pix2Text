# coding: utf-8
# Copyright (C) 2022-2024, [Breezedeus](https://github.com/breezedeus).
#
# credit to: pix2tex, lukas-blecher/LaTeX-OCR
# Adapted from https://github.com/lukas-blecher/LaTeX-OCR/blob/main/pix2tex/cli.py
from typing import Tuple, Optional, Dict, Any, Union
import logging
import yaml
from pathlib import Path

from PIL import Image
from transformers import PreTrainedTokenizerFast
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
from cnstd.utils import get_model_file

from pix2tex.utils import *
from pix2tex.models import get_model
from pix2tex.dataset.transforms import test_transform

from .consts import LATEX_CONFIG_FP, MODEL_VERSION, AVAILABLE_MODELS, DOWNLOAD_SOURCE
from .utils import data_dir


logger = logging.getLogger(__name__)


def download_checkpoints(model_info, resizer_model_info, args):
    # adapted from pix2tex.model.checkpoints.get_latest_checkpoint
    ckpt_list = [args.mfr_checkpoint, args.resizer_checkpoint]
    for info, fp in zip([model_info, resizer_model_info], ckpt_list):
        if not os.path.exists(fp):
            logger.warning(f'model file {fp} not found, downloading...')
            info['filename'] = info['url']
            get_model_file(
                info, args.model_root_dir, download_source=DOWNLOAD_SOURCE
            )  # download the .zip file and unzip


def minmax_size(
    img: Image,
    max_dimensions: Tuple[int, int] = None,
    min_dimensions: Tuple[int, int] = None,
) -> Image:
    """Resize or pad an image to fit into given dimensions

    Args:
        img (Image): Image to scale up/down.
        max_dimensions (Tuple[int, int], optional): Maximum dimensions. Defaults to None.
        min_dimensions (Tuple[int, int], optional): Minimum dimensions. Defaults to None.

    Returns:
        Image: Image with correct dimensionality
    """
    if max_dimensions is not None:
        ratios = [a / b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size) // max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
        padded_size = [
            max(img_dim, min_dim) for img_dim, min_dim in zip(img.size, min_dimensions)
        ]
        if padded_size != list(img.size):  # assert hypothesis
            padded_im = Image.new('L', padded_size, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img


def find_all_left_or_right(latex, left_or_right='left'):
    left_bracket_infos = []
    prefix_len = len(left_or_right) + 1
    # 匹配出latex中所有的 '\left' 后面跟着的第一个非空格字符，定位它们所在的位置
    for m in re.finditer(rf'\\{left_or_right}\s*\S', latex):
        start, end = m.span()
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


def post_post_process_latex(latex: str) -> str:
    """对识别结果做进一步处理和修正。"""
    # 把latex中的中文括号全部替换成英文括号
    latex = latex.replace('（', '(').replace('）', ')')
    # 把latex中的中文逗号全部替换成英文逗号
    latex = latex.replace('，', ',')

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
    return latex


class LatexOCR(object):
    """Get a prediction of an image in the easiest way"""

    image_resizer = None
    last_pic = None

    @in_model_path()
    def __init__(
        self,
        *,
        model_name: str = 'mfr',
        model_backend: str = 'pytorch',
        context: str = 'cpu',  # ['cpu', 'gpu', 'cuda']
        model_fp: Optional[Union[str, Path]] = None,
        root: Union[str, Path] = data_dir(),
        arguments: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a LatexOCR model

        Args:
            model_name (str, optional): The name of the model. Defaults to 'mfr'.
            model_backend (str, optional): The model backend. Defaults to 'pytorch'.
            context (str, optional): What device to use for computation, supports `['cpu', 'cuda', 'gpu']`; defaults to `cpu`.
            model_fp (Optional[Union[str, Path]], optional): The model file path. Defaults to None.
            root (Union[str, Path], optional): The model root directory. Defaults to data_dir().
            arguments (Union[Namespace, Munch], optional): Special model parameters. Defaults to None.
        """
        model_root_dir = Path(root) / MODEL_VERSION / 'formula'
        model_info = AVAILABLE_MODELS.get_info(model_name, model_backend)
        resizer_model_info = AVAILABLE_MODELS.get_info('resizer', 'pytorch')

        def_arguments = {
            'config': LATEX_CONFIG_FP,
            'mfr_checkpoint': model_root_dir / model_info['fn'],
            'resizer_checkpoint': model_root_dir / resizer_model_info['fn'],
            'model_root_dir': model_root_dir,
            # 'no_cuda': True,
            'no_resize': False,
            'device': context,
        }
        if arguments is not None:
            if 'model_fp' in arguments:
                arguments['mfr_checkpoint'] = arguments.pop('model_fp')
            def_arguments.update(arguments)
        if model_fp is not None:
            def_arguments['mfr_checkpoint'] = model_fp
        arguments = def_arguments

        arguments = Munch(arguments)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        with open(arguments.config, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.args = parse_args(Munch(params))
        self.args.update(**vars(arguments))
        self.args.wandb = False
        # self.args.device = (
        #     'cuda' if torch.cuda.is_available() and not self.args.no_cuda else 'cpu'
        # )
        download_checkpoints(model_info, resizer_model_info, self.args)

        self.model = get_model(self.args)
        self.model.load_state_dict(
            torch.load(self.args.mfr_checkpoint, map_location=self.args.device)
        )
        logger.info(f'use model: {self.args.mfr_checkpoint}')
        self.model.eval()

        if not self.args.no_resize and os.path.isfile(self.args.resizer_checkpoint):
            self.image_resizer = ResNetV2(
                layers=[2, 3, 3],
                num_classes=max(self.args.max_dimensions) // 32,
                global_pool='avg',
                in_chans=1,
                drop_rate=0.05,
                preact=True,
                stem_type='same',
                conv_layer=StdConv2dSame,
            ).to(self.args.device)
            self.image_resizer.load_state_dict(
                torch.load(self.args.resizer_checkpoint, map_location=self.args.device)
            )
            logger.info(f'use model: {self.args.resizer_checkpoint}')
            self.image_resizer.eval()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.args.tokenizer)

    @in_model_path()
    def __call__(self, img=None, resize=True) -> str:
        """Get a prediction from an image

        Args:
            img (Image, optional): Image to predict. Defaults to None.
            resize (bool, optional): Whether to call the resize model. Defaults to True.

        Returns:
            str: predicted Latex code
        """
        if type(img) is bool:
            img = None
        if img is None:
            if self.last_pic is None:
                print('Provide an image.')
                return ''
            else:
                img = self.last_pic.copy()
        else:
            self.last_pic = img.copy()
        img = minmax_size(pad(img), self.args.max_dimensions, self.args.min_dimensions)
        if (self.image_resizer is not None and not self.args.no_resize) and resize:
            with torch.no_grad():
                input_image = img.convert('RGB').copy()
                r, w, h = 1, input_image.size[0], input_image.size[1]
                for _ in range(10):
                    h = int(h * r)  # height to resize
                    try:
                        img = pad(
                            minmax_size(
                                input_image.resize(
                                    (w, h),
                                    Image.Resampling.BILINEAR
                                    if r > 1
                                    else Image.Resampling.LANCZOS,
                                ),
                                self.args.max_dimensions,
                                self.args.min_dimensions,
                            )
                        )
                        t = test_transform(image=np.array(img.convert('RGB')))['image'][
                            :1
                        ].unsqueeze(0)
                        w = (
                            self.image_resizer(t.to(self.args.device)).argmax(-1).item()
                            + 1
                        ) * 32
                        logger.debug((r, img.size, (w, int(input_image.size[1] * r))))
                    except Exception as e:
                        logger.warning(e)
                        break

                    if w == img.size[0]:
                        break
                    r = w / img.size[0]
        else:
            img = np.array(pad(img).convert('RGB'))
            t = test_transform(image=img)['image'][:1].unsqueeze(0)
        im = t.to(self.args.device)

        dec = self.model.generate(
            im.to(self.args.device), temperature=self.args.get('temperature', 0.25)
        )
        pred = post_process(token2str(dec, self.tokenizer)[0])
        pred = post_post_process_latex(pred)
        return pred
