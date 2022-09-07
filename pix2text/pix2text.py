# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import logging

from PIL import Image
from cnocr import CnOcr, ImageClassifier

from .latex_ocr import LatexOCR

logger = logging.getLogger(__name__)


class Pix2Text(object):
    categories = ('text', 'english', 'formula')
    # model_fp = './data/image-formula-text/image-clf-epoch=015-val-accuracy-epoch=0.9394-model.ckpt'

    def __init__(self, clf_model_fp):
        self.LATEX_MODEL = LatexOCR()
        transform_configs = {
            'crop_size': [150, 450],
            'resize_size': 160,
            'resize_max_size': 1000,
        }
        self.image_clf = ImageClassifier(
            base_model_name='mobilenet_v2',
            categories=self.categories,
            transform_configs=transform_configs,
        )
        self.image_clf.load(clf_model_fp, 'cpu')

        self.general_ocr = CnOcr()
        self.english_ocr = CnOcr(
            det_model_name='en_PP-OCRv3_det', rec_model_name='en_PP-OCRv3'
        )

    def __call__(self, img):
        return self.extract(img)

    def extract(self, img):
        res = self.image_clf.predict_images([img])[0]
        logger.info('CLF Result: %s', res)
        image_type = res[0]
        if res[1] < 0.65 and res[0] == 'formula':
            image_type = 'text'
        if res[1] < 0.75 and res[0] == 'english':
            image_type = 'text'
        if image_type == 'formula':
            result = self.latex(img)
        else:
            result = self.ocr(img, image_type)

        return image_type, result

    def ocr(self, image, image_type):
        ocr_model = self.english_ocr if image_type == 'english' else self.general_ocr
        result = ocr_model.ocr(image)
        texts = [_one['text'] for _one in result]
        # logger.info(f'\tOCR results: {pformat(texts)}\n')
        result = '\n'.join(texts)
        return result

    def latex(self, image):
        out = self.latex_model(Image.open(image))
        return str(out)
