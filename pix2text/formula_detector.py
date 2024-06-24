# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).
from typing import Optional, Union, Tuple
from pathlib import Path
import logging

from cnstd.yolo_detector import YoloDetector

from .consts import AVAILABLE_MODELS
from .utils import data_dir, prepare_model_files

logger = logging.getLogger(__name__)


BACKEND_TO_EXTENSION_MAPPING = {
    'pytorch': 'pt',
    'onnx': 'onnx',
    'coreml': 'mlpackage',
    'torchscript': 'torchscript',
}


class MathFormulaDetector(YoloDetector):
    def __init__(
        self,
        *,
        model_name: str = 'mfd',
        model_backend: str = 'onnx',
        device: Optional[str] = None,
        model_path: Optional[Union[str, Path]] = None,
        root: Union[str, Path] = data_dir(),
        static_resized_shape: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs,
    ):
        """
        Math Formula Detector based on YOLO.

        Args:
            model_name (str): model name, default is 'mfd'.
            model_backend (str): model backend, default is 'onnx'.
            device (optional str): device to use, default is None.
            model_path (optional str): model path, default is None.
            root (optional str): root directory to save model files, default is data_dir().
            static_resized_shape (optional int or tuple): static resized shape, default is None.
                When it is not None, the input image will be resized to this shape before detection,
                ignoring the input parameter `resized_shape` if .detect() is called.
                Some format of models may require a fixed input size, such as CoreML.
            **kwargs (): other parameters.
        """
        if model_path is None:
            model_info = AVAILABLE_MODELS.get_info(model_name, model_backend)
            model_path = prepare_model_files(root, model_info)
            extension = BACKEND_TO_EXTENSION_MAPPING.get(model_backend, model_backend)
            cand_paths = find_files(model_path, f'.{extension}')
            if not cand_paths:
                raise FileNotFoundError(f'can not find available file in {model_path}')
            model_path = cand_paths[0]
        logger.info(f'Use model path for MFD: {model_path}')

        super().__init__(
            model_path=model_path,
            device=device,
            static_resized_shape=static_resized_shape,
            **kwargs,
        )


def find_files(directory, extension):
    # 创建Path对象
    dir_path = Path(directory)

    pattern = f"mfd*{extension}"

    outs = []
    # 使用rglob方法递归查找匹配的文件
    for file_path in dir_path.rglob(pattern):
        # 检查文件名是否不以点开头（除了文件扩展名）
        if not file_path.name.startswith('.') or file_path.suffix == extension:
            outs.append(file_path)

    return outs
