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



返回结果 `out_text` 是个 `dict`，其中 key `image_type` 表示图片分类类别，而 key `text` 表示识别的结果。



以下是一些示例图片的识别结果：

<table>
<tr>
<td> 图片 </td> <td> Pix2Text识别结果 </td>
</tr>
<tr>
<td>

<img src="./docs/examples/formula.jpg" alt="formula"> 
</td>
<td>

```json
{"image_type": "formula",
 "text": "\\mathcal{L}_{\\mathrm{eyelid}}~\\longrightarrow"
 "\\sum_{t=1}^{T}\\sum_{v=1}^{V}\\mathcal{N}"
 "\\cal{M}_{v}^{\\mathrm{(eyelid}})"
 "\\left(\\left|\\left|\\hat{h}_{t,v}\\,-\\,"
 "\\mathcal{x}_{t,v}\\right|\\right|^{2}\\right)"}
```
</td>
</tr>
<tr>
<td>

 <img src="./docs/examples/english.jpg" alt="english"> 
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

 <img src="./docs/examples/general.jpg" alt="general"> 
</td>
<td>

```json
{"image_type": "general",
 "text": "618\n开门红提前购\n很贵\n买贵返差\n终于降价了\n"
          "100%桑蚕丝\n要买趁早\n今日下单188元\n仅限一天"}
```
</td>
</tr>
</table>







## 安装

嗯，顺利的话一行命令即可。

```bash
pip install pix2text
```

安装速度慢的话，可以指定国内的安装源，如使用豆瓣源：

```bash
pip install pix2text -i https://pypi.doubanio.com/simple
```



如果是初次使用**OpenCV**，那估计安装都不会很顺利，bless。

**Pix2Text** 主要依赖 [**CnOCR>=2.2.2**](https://github.com/breezedeus/cnocr) ，以及 [**LaTeX-OCR**](https://github.com/lukas-blecher/LaTeX-OCR) 。如果安装过程遇到问题，也可参考它们的安装说明文档。



> **Warning** 
>
> 如果电脑中从未安装过 `PyTorch`，`OpenCV` python包，初次安装可能会遇到不少问题，但一般都是常见问题，可以自行百度/Google解决。



## HTTP服务

 **Pix2Text** 加入了基于 FastAPI 的HTTP服务。开启服务需要安装几个额外的包，可以使用以下命令安装：

```bash
pip install pix2text[serve]
```



安装完成后，可以通过以下命令启动HTTP服务（**`-p`** 后面的数字是**端口**，可以根据需要自行调整）：

```bash
p2t serve -p 8503
```



服务开启后，可以使用以下方式调用服务。



### 命令行

比如待识别文件为 `docs/examples/english.jpg`，如下使用 curl 调用服务：

```bash
> curl -F image=@docs/examples/english.jpg http://0.0.0.0:8503/pix2text
```



### Python

使用如下方式调用服务：

```python
import requests

image_fp = 'docs/examples/english.jpg'
r = requests.post(
    'http://0.0.0.0:8503/pix2text', files={'image': (image_fp, open(image_fp, 'rb'), 'image/png')},
)
out = r.json()['results']
print(out)
```



### 其他语言

请参照 curl 的调用方式自行实现。
