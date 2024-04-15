# 常见问题（FAQ）

## CnOCR 是免费的吗？

CnOCR 代码是免费的，而且是开源的。可以按需自行调整发布或商业使用。

## CnOCR 能识别英文以及空格吗？

可以。

## CnOCR 能识别繁体中文吗？

部分模型可以，具体见 [可用模型](models.md)。

## CnOCR 能识别竖排文字的图片吗？

部分模型可以，具体见 [可用模型](models.md)。

## CnOCR 能支持其他语言的模型吗？

暂时没有。如有其他外语（如日、韩等）识别需求，可在 **知识星球** [**CnOCR/CnSTD私享群**](https://t.zsxq.com/FEYZRJQ) 中向作者提出建议。



## 不同机器上使用同样的模型预测结果不同

很大可能是因为不同运行环境下的 **Pillow包 **版本不同（查看版本号：`pip show pillow`），请把预测环境的 Pillow 版本统一到训练时使用的版本。CnOCR中会使用 Pillow 的  `Image.open()` 读入图片，不同版本的 Pillow 调用  `Image.open()` 时可能得到不同的取值。

同时，尽量保证训练和预测使用的 Python 环境是使用相同方式安装的，因为有人发现用 `pip` 和 `conda` 安装的相同版本的 Pillow，也可能导致不同的结果。具体参考：

* [Same image has different values once imported, same OS, python version and pillow version but install pip vs conda · Issue #5887 · python-pillow/Pillow](https://github.com/python-pillow/Pillow/issues/5887)



## 文本检测的部分结果翻转了180度

CnOCR 中已支持**角度判断功能**，可通过开启此功能来修正检测文本翻转180度的问题。`CnOcr` 初始化时传入以下参数即可开启角度判断功能。

```python
from cnocr import CnOcr

img_fp = './docs/examples/huochepiao.jpeg'
ocr = CnOcr(det_more_configs={'use_angle_clf': True})  # 开启角度判断功能
out = ocr.ocr(img_fp)

print(out)
```

具体可参考 [CnSTD 文档](https://github.com/breezedeus/cnstd) 。
