# coding: utf-8
from pathlib import Path
from typing import Union

from PIL import Image
from cnstd import LayoutAnalyzer
from cnstd.yolov7.consts import CATEGORY_DICT

from .utils import read_img, save_layout_img


class LayoutParser(object):
    def __init__(
        self,
        model_type: str = 'yolov7_tiny',  # 当前仅支持 `yolov7_tiny`
        model_backend: str = 'pytorch',
        device: str = 'cpu',
        **kwargs
    ):
        self.layout_model = LayoutAnalyzer(
            model_name='layout',
            model_type=model_type,
            model_backend=model_backend,
            device=device,
            **kwargs
        )
        self.ignored_types = {'_background_', 'Footer'}
        self.type_mappings = {
            'Header': 'Text',
            'Text': 'Text',
            'Title': 'Title',
            'Figure': 'Figure',
            'Figure caption': 'Text',
            'Table': 'Table',
            'Table caption': 'Text',
            'Reference': 'Text',
            'Equation': 'Formula',
        }

    def __call__(self, *args, **kwargs):
        return self.parse(*args, **kwargs)

    def parse(self, img: Union[str, Path, Image.Image], **kwargs):
        if isinstance(img, Image.Image):
            img0 = img.convert('RGB')
        else:
            img0 = read_img(img, return_type='Image')
        resized_shape = kwargs.get('resized_shape', 608)
        layout_out = self.layout_model(img0.copy(), resized_shape=resized_shape)

        if kwargs.get('save_analysis_res'):
            save_layout_img(
                img0,
                CATEGORY_DICT['layout'],
                layout_out,
                kwargs.get('save_analysis_res'),
                key='box',
            )

        final_out = []
        for box_info in layout_out:
            image_type = box_info['type']
            if image_type in self.ignored_types:
                continue
            image_type = self.type_mappings.get(image_type, image_type)
            final_out.append({'type': image_type, 'position': box_info['box']})

        return final_out
