# coding: utf-8
from enum import Enum
from pathlib import Path
from typing import Union, Optional, List, Dict, Any

from PIL import Image
from cnstd import LayoutAnalyzer
from cnstd.yolov7.consts import CATEGORY_DICT

from .utils import read_img, save_layout_img, select_device


class ElementType(Enum):
    ABANDONED = -2  # 可以指定有些区域不做识别，如 Image 与 Image caption 中间地带
    IGNORED = -1
    UNKNOWN = 0
    TEXT = 1
    TITLE = 2
    FIGURE = 3
    TABLE = 4
    FORMULA = 5
    PLAIN_TEXT = 11  # 与 TEXT 类似，但是绝对不包含公式

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

class LayoutParser(object):
    def __init__(
        self,
        model_type: str = 'yolov7_tiny',  # 当前仅支持 `yolov7_tiny`
        model_backend: str = 'pytorch',  # 当前仅支持 `pytorch`
        device: str = None,
        **kwargs
    ):
        device = select_device(device)
        device = device if device != 'mps' else 'cpu'
        self.layout_model = LayoutAnalyzer(
            model_name='layout',
            model_type=model_type,
            model_backend=model_backend,
            device=device,
            **kwargs,
        )
        self.ignored_types = {'_background_', 'Footer', 'Reference'}
        self.type_mappings = {
            'Header': ElementType.TEXT,
            'Text': ElementType.TEXT,
            'Title': ElementType.TITLE,
            'Figure': ElementType.FIGURE,
            'Figure caption': ElementType.TEXT,
            'Table': ElementType.TABLE,
            'Table caption': ElementType.TEXT,
            'Reference': ElementType.TEXT,
            'Equation': ElementType.FORMULA,
        }

    @classmethod
    def from_config(cls, configs: Optional[dict] = None, device: str = None, **kwargs):
        configs = configs or {}
        device = select_device(device)
        configs['device'] = device if device != 'mps' else 'cpu'

        return cls(
            model_type=configs.get('model_type', 'yolov7_tiny'),
            model_backend=configs.get('model_backend', 'pytorch'),
            device=device,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        return self.parse(*args, **kwargs)

    def parse(
        self,
        img: Union[str, Path, Image.Image],
        resized_shape: int = 608,
        table_as_image: bool = False,
        **kwargs
    ) -> (List[Dict[str, Any]], Dict[str, Any]):
        """

        Args:
            img ():
            resized_shape ():
            table_as_image ():
            **kwargs ():

        Returns: parsed results & column meta information;
            the parsed results is a list of dict with keys: 'type', 'position', 'score':
                 * type: ElementType
                 * position: np.ndarray, with shape of (4, 2)
                 * score: float
            the column meta is a dict, with column number as its keys.

        """
        if isinstance(img, Image.Image):
            img0 = img.convert('RGB')
        else:
            img0 = read_img(img, return_type='Image')
        layout_out = self.layout_model(img0.copy(), resized_shape=resized_shape)

        if kwargs.get('save_layout_res'):
            save_layout_img(
                img0,
                CATEGORY_DICT['layout'],
                layout_out,
                kwargs.get('save_layout_res'),
                key='box',
            )

        final_out = []
        for box_info in layout_out:
            image_type = box_info['type']
            if image_type in self.ignored_types:
                continue
            image_type = self.type_mappings.get(image_type, image_type)
            if table_as_image and image_type == ElementType.TABLE:
                image_type = ElementType.FIGURE
            final_out.append(
                {
                    'type': image_type,
                    'position': box_info['box'],
                    'score': box_info['score'],
                }
            )

        return final_out, {}
