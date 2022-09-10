<div align="center">
  <img src="./docs/figs/p2t.jpg" width="250px"/>
  <div>&nbsp;</div>

[![license](https://img.shields.io/github/license/breezedeus/pix2text)](./LICENSE)
[![PyPI version](https://badge.fury.io/py/pix2text.svg)](https://badge.fury.io/py/pix2text)
[![forks](https://img.shields.io/github/forks/breezedeus/pix2text)](https://github.com/breezedeus/pix2text)
[![stars](https://img.shields.io/github/stars/breezedeus/pix2text)](https://github.com/breezedeus/pix2text)
![last-release](https://img.shields.io/github/release-date/breezedeus/pix2text)
![last-commit](https://img.shields.io/github/last-commit/breezedeus/pix2text)
[![Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fbreezedeus)](https://twitter.com/breezedeus)

[🛀🏻 在线Demo](https://huggingface.co/spaces/breezedeus/cnocr) |
[💬 交流群](https://cnocr.readthedocs.io/zh/latest/contact/)

</div>

<div align="center">
[English](./README_en.md) | 中文
</div>

# Pix2Text



**Pix2Text** 期望成为 **[Mathpix](https://mathpix.com/)** 的**免费开源 Python **替代工具，完成与 Mathpix 类似的功能。当前 Pix2Text 可识别截屏图片中的**数学公式**、**英文**、或者**中文文字**。它的流程如下：

<div align="center">
  <img src="./docs/figs/arch-flow.jpg" alt="Pix2Text流程" width="800px"/>
</div>



Pix2Text首先利用**图片分类模型**来判断图片类型，然后基于不同的图片类型，把图片交由不同的识别系统进行文字识别：

1. 如果图片类型为 `formula` ，表示图片为数学公式，此时调用 [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR) 识别图片中的数学公式，返回其Latex表示；
1. 如果图片类型为`english`，表示图片中包含的是英文文字，此时使用 [CnOCR](https://github.com/breezedeus/cnocr) 中的**英文模型**识别其中的英文文字；英文模型对于纯英文的文字截图，识别效果比通用模型好；
1. 如果图片类型为`general`，表示图片中包含的是常见文字，此时使用 [CnOCR](https://github.com/breezedeus/cnocr) 中的**通用模型**识别其中的中或英文文字。



后续图片类型会依据应用需要做进一步的细分。





## 使用说明

安装好后，调用很简单，以下是示例：

```python
from pix2text import Pix2Text

img_fp = './docs/examples/formula.jpg'
p2t = Pix2Text()
out_text = p2t(img_fp)  # 也可以使用 `p2t.recognize(img_fp)` 获得相同的结果
print(out_text)
```



其中返回结果 `out_text` 是个 `dict`，其中 key `image_type` 表示图片分类类别，而 key `text` 表示识别的结果。如上面的调用返回以下结果：

<table>
<tr>
<td> 图片 </td> <td> Pix2Text识别结果 </td>
</tr>
<tr>
<td>

<img src="./docs/examples/formula.jpg" alt="formula"  width="30%"> 
</td>
<td>

```json
{"image_type": "formula",
 "text": "\\mathcal{L}_{\\mathrm{eyelid}}~\\longrightarrow"
 "\\sum_{t=1}^{T}\\sum_{v=1}^{V}\\mathcal{N}"
 "\\cal{M}_{v}^{\\mathrm{(eyelid}})"
 "\\left(\\left|\\left|\\hat{h}_{t,v}\\,-\\,\\mathcal{x}_{t,v}"
 "\\right|\\right|^{2}\\right)"}
```
</td>
</tr>
<tr>
<td>

 <img src="./docs/examples/english.jpg" alt="english"  width="30%"> 
</td>
<td>

```json
{"image_type": "english",
 "text": "python scripts/screenshot_daemon_with_server\n"
         "2-get_model:178usemodel:/Users/king/.cr\n"
         "enet_lite_136-fc-epoch=039-complete_match_er"}
```
</td>
</tr>
<tr>
<td>

 <img src="./docs/examples/general.jpg" alt="general" width="30%"> 
</td>
<td>

```json
{"image_type": "general",
 "text": "618\n开门红提前购\n很贵\n买贵返差\n终于降价了"
          "\n100%桑蚕丝\n要买趁早\n今日下单188元\n仅限一天"}
```
</td>
</tr>
</table>







## Install



