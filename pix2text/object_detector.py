# coding: utf-8

from typing import List, Dict, Any

import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import (
    fcos_resnet50_fpn,
    FCOS_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
    ssd300_vgg16,
    SSD300_VGG16_Weights,
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)

from .utils import read_img
from .category_mapping import CATEGORY_MAPPINGS


MODELS = {
    "frcnn-resnet": (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    ),  # 7s
    "frcnn-mobilenet": (  # 1s
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
    ),
    "retinanet": (
        retinanet_resnet50_fpn_v2,
        RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
    ),  # 3s
    "ssd300_vgg16": (ssd300_vgg16, SSD300_VGG16_Weights.DEFAULT),
    "ssdlite320_mobilenet_v3_large": (  # <1s
        ssdlite320_mobilenet_v3_large,
        SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
    ),
    "fcos_resnet50_fpn": (fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights.DEFAULT),  # 3s
}


class ObjectDetector(object):
    def __init__(
        self, model_name='frcnn-mobilenet', score_thresh=0.6, nms_thresh=0.2, **kwargs
    ):
        model_cls, weights = MODELS[model_name]
        cls_configs = dict(
            score_thresh=score_thresh,
            box_score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            box_nms_thresh=nms_thresh,
        )
        self.model = model_cls(weights=weights, **cls_configs)
        self.model.eval()
        self.transform = weights.transforms()
        self.categories = weights.meta["categories"]

    @torch.no_grad()
    def __call__(self, img) -> List[Dict[str, Any]]:
        batch = [self.transform(img)]
        prediction = self.model(batch)[0]
        labels = [
            CATEGORY_MAPPINGS.get(self.categories[i], self.categories[i])
            for i in prediction["labels"]
        ]
        results = []
        for l, s, b in zip(labels, prediction['scores'], prediction['boxes']):
            results.append(({'text': l, 'score': s, 'position': b}))

        return results


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    # import requests
    # response = requests.get('https://i.ytimg.com/vi/q71MCWAEfL8/maxresdefault.jpg')
    # open("obj_det.jpeg", "wb").write(response.content)
    img = read_img("examples/00810-good-0.8948.jpg")
    detector = ObjectDetector(model_name='frcnn-mobilenet')

    start_time = time.time()
    out = detector(img)
    print(f'time cost: {time.time() - start_time}')

    print(out)
    labels = [_one['pred'] for _one in out]
    boxes = [_one['box'] for _one in out]

    colors = (
        np.random.uniform(0, 255, size=(len(detector.categories), 3))
        .astype('int')
        .tolist()
    )
    colors = [tuple(c) for c in colors]

    box = draw_bounding_boxes(
        img,
        boxes=torch.stack(boxes, dim=0),
        labels=labels,
        colors=colors,
        width=2,
        font_size=30,
        font='Arial',
    )

    im = to_pil_image(box.detach())

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(im)
    plt.show()
