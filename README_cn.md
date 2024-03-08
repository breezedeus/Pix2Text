<div align="center">
  <img src="./docs/figs/p2t-logo.png" width="250px"/>
  <div>&nbsp;</div>

[![Discord](https://img.shields.io/discord/1200765964434821260?label=Discord)](https://discord.gg/drT8H85Y)
[![Downloads](https://static.pepy.tech/personalized-badge/pix2text?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/pix2text)
[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fbreezedeus%2FPix2Text&label=Visitors&countColor=%23ff8a65&style=flat&labelStyle=none)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2Fbreezedeus%2FPix2Text)
[![license](https://img.shields.io/github/license/breezedeus/pix2text)](./LICENSE)
[![PyPI version](https://badge.fury.io/py/pix2text.svg)](https://badge.fury.io/py/pix2text)
[![forks](https://img.shields.io/github/forks/breezedeus/pix2text)](https://github.com/breezedeus/pix2text)
[![stars](https://img.shields.io/github/stars/breezedeus/pix2text)](https://github.com/breezedeus/pix2text)
![last-release](https://img.shields.io/github/release-date/breezedeus/pix2text)
![last-commit](https://img.shields.io/github/last-commit/breezedeus/pix2text)
[![Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fbreezedeus)](https://twitter.com/breezedeus)

[👩🏻‍💻 网页版](https://p2t.breezedeus.com) |
[👨🏻‍💻 在线 Demo](https://huggingface.co/spaces/breezedeus/Pix2Text-Demo) |
[💬 交流群](https://www.breezedeus.com/join-group)

</div>

<div align="center">

[English](./README.md) | 中文


</div>

# Pix2Text (P2T)

## Update 2024.02.26：发布 **V1.0**

主要变更：

* 数学公式识别（MFR）模型使用新架构，在新的数据集上训练，获得了 SOTA 的精度。具体说明请见：[Pix2Text V1.0 新版发布：最好的开源公式识别模型 | Breezedeus.com](https://www.breezedeus.com/article/p2t-v1.0)。

## Update 2024.01.10：发布 V0.3

主要变更：

* 支持识别 **`80+` 种语言**，详细语言列表见 [支持的语言列表](#支持的语言列表)；
* 模型自动下载增加国内站点；
* 优化对检测 boxes 的合并逻辑。


## Update 2023.07.03：发布 V0.2.3

主要变更：
* 训练了新的**公式识别模型**，供 **[P2T网页版](https://p2t.breezedeus.com)** 使用。新模型精度更高，尤其对**手写公式**和**多行公式**类图片。具体参考：[Pix2Text 新版公式识别模型 | Breezedeus.com](https://www.breezedeus.com/article/p2t-mfd-20230702) 。
* 优化了对检测出的boxes的排序逻辑，以及对混合图片的处理逻辑，使得最终识别效果更符合直觉。
* 优化了识别结果的合并逻辑，自动判断是否该换行，是否分段。


了解更多：[RELEASE.md](./RELEASE.md) 。

---



**Pix2Text (P2T)** 期望成为 **[Mathpix](https://mathpix.com/)** 的**免费开源 Python **替代工具，目前已经可以完成 **Mathpix** 的核心功能。**Pix2Text (P2T)** 自 **V0.2** 开始，支持识别**既包含文字又包含公式的混合图片**，返回效果类似于 **Mathpix**。P2T 的核心原理见下图（文字识别支持**中文**和**英文**）：

<div align="center">
  <img src="./docs/figs/arch-flow2.jpg" alt="Pix2Text流程" width="600px"/>
</div>


**P2T** 使用开源工具  **[CnSTD](https://github.com/breezedeus/cnstd)** 检测出图片中**数学公式**所在位置，再交由 **P2T** 自己的**公式识别引擎（LatexOCR）** 识别出各对应位置数学公式的Latex表示。图片的剩余部分再交由 **文字识别引擎（[CnOCR](https://github.com/breezedeus/cnocr) 或 [EasyOCR](https://github.com/JaidedAI/EasyOCR)）** 进行文字检测和文字识别。最后 **P2T** 合并所有识别结果，获得最终的图片识别结果。感谢这些开源工具。



P2T 作为Python3工具包，对于不熟悉Python的朋友不太友好，所以我们也发布了**可免费使用**的 **[P2T网页版](https://p2t.breezedeus.com)**，直接把图片丢进网页就能输出P2T的解析结果。**网页版会使用最新的模型，效果会比开源模型更好。**



感兴趣的朋友欢迎扫码加小助手为好友，备注 `p2t`，小助手会定期统一邀请大家入群。群内会发布P2T相关工具的最新进展：

<div align="center">
  <img src="./docs/figs/wx-qr-code.JPG" alt="微信群二维码" width="300px"/>
</div>



作者也维护 **知识星球** [**P2T/CnOCR/CnSTD私享群**](https://t.zsxq.com/FEYZRJQ) ，这里面的提问会较快得到作者的回复，欢迎加入。**知识星球私享群**也会陆续发布一些P2T/CnOCR/CnSTD相关的私有资料，包括**部分未公开的模型**，**购买付费模型享优惠**，**不同应用场景的调用代码**，使用过程中遇到的难题解答等。星球也会发布P2T/OCR/STD相关的最新研究资料。



## 支持的语言列表

Pix2Text 的文字识别引擎支持 **`80+` 种语言**，如**英文、简体中文、繁体中文、越南语**等。其中，**英文**和**简体中文**识别使用的是开源 OCR 工具 **[CnOCR](https://github.com/breezedeus/cnocr)** ，其他语言的识别使用的是开源 OCR 工具 **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** ，感谢相关的作者们。

支持的**语言列表**和**语言代码**如下：
<details>
<summary>↓↓↓ Click to show details ↓↓↓</summary>


| Language            | Code Name   |
| ------------------- | ----------- |
| Abaza               | abq         |
| Adyghe              | ady         |
| Afrikaans           | af          |
| Angika              | ang         |
| Arabic              | ar          |
| Assamese            | as          |
| Avar                | ava         |
| Azerbaijani         | az          |
| Belarusian          | be          |
| Bulgarian           | bg          |
| Bihari              | bh          |
| Bhojpuri            | bho         |
| Bengali             | bn          |
| Bosnian             | bs          |
| Simplified Chinese  | ch_sim      |
| Traditional Chinese | ch_tra      |
| Chechen             | che         |
| Czech               | cs          |
| Welsh               | cy          |
| Danish              | da          |
| Dargwa              | dar         |
| German              | de          |
| English             | en          |
| Spanish             | es          |
| Estonian            | et          |
| Persian (Farsi)     | fa          |
| French              | fr          |
| Irish               | ga          |
| Goan Konkani        | gom         |
| Hindi               | hi          |
| Croatian            | hr          |
| Hungarian           | hu          |
| Indonesian          | id          |
| Ingush              | inh         |
| Icelandic           | is          |
| Italian             | it          |
| Japanese            | ja          |
| Kabardian           | kbd         |
| Kannada             | kn          |
| Korean              | ko          |
| Kurdish             | ku          |
| Latin               | la          |
| Lak                 | lbe         |
| Lezghian            | lez         |
| Lithuanian          | lt          |
| Latvian             | lv          |
| Magahi              | mah         |
| Maithili            | mai         |
| Maori               | mi          |
| Mongolian           | mn          |
| Marathi             | mr          |
| Malay               | ms          |
| Maltese             | mt          |
| Nepali              | ne          |
| Newari              | new         |
| Dutch               | nl          |
| Norwegian           | no          |
| Occitan             | oc          |
| Pali                | pi          |
| Polish              | pl          |
| Portuguese          | pt          |
| Romanian            | ro          |
| Russian             | ru          |
| Serbian (cyrillic)  | rs_cyrillic |
| Serbian (latin)     | rs_latin    |
| Nagpuri             | sck         |
| Slovak              | sk          |
| Slovenian           | sl          |
| Albanian            | sq          |
| Swedish             | sv          |
| Swahili             | sw          |
| Tamil               | ta          |
| Tabassaran          | tab         |
| Telugu              | te          |
| Thai                | th          |
| Tajik               | tjk         |
| Tagalog             | tl          |
| Turkish             | tr          |
| Uyghur              | ug          |
| Ukranian            | uk          |
| Urdu                | ur          |
| Uzbek               | uz          |
| Vietnamese          | vi          |


> Ref: [Supported Languages](https://www.jaided.ai/easyocr/) .

</details>



## P2T 网页版

所有人都可以免费使用 **[P2T网页版](https://p2t.breezedeus.com)**，每人每天可以免费识别 10000 个字符，正常使用应该够用了。*请不要批量调用接口，机器资源有限，批量调用会导致其他人无法使用服务。*

受限于机器资源，网页版当前只支持**简体中文和英文**，要尝试其他语言上的效果，请使用以下的**在线 Demo**。



## 在线 Demo 🤗

也可以使用 **[在线 Demo](https://huggingface.co/spaces/breezedeus/Pix2Text-Demo)** 尝试 **P2T** 在不同语言上的效果。但在线 Demo 使用的硬件配置较低，速度会较慢。如果是简体中文或者英文图片，建议使用 **[P2T网页版](https://p2t.breezedeus.com)**。



## 使用说明

### 识别既有公式又有文本的混合图片

对于既有公式又有文本的混合图片，使用函数 `.recognize()` 识别图片中的文字和数学公式。如针对以下图片 ([docs/examples/en1.jpg](docs/examples/en1.jpg))：

<div align="center">
  <img src="./docs/examples/en1.jpg" alt="English mixed image" width="600px"/>
</div>

调用方式如下：

```python
from pix2text import Pix2Text, merge_line_texts

img_fp = './docs/examples/en1.jpg'
p2t = Pix2Text()
outs = p2t.recognize(img_fp, resized_shape=608)  # 也可以使用 `p2t(img_fp)` 获得相同的结果
print(outs)
# 如果只需要识别出的文字和Latex表示，可以使用下面行的代码合并所有结果
only_text = merge_line_texts(outs, auto_line_break=True)
print(only_text)
```

返回结果 `outs` 是个 `dict`，其中 key `position` 表示Box位置信息，`type` 表示类别信息，而 `text` 表示识别的结果。具体说明见[接口说明](#接口说明)。



### 识别纯公式图片

对于只包含数学公式的图片，使用函数 `.recognize_formula()` 可以把数学公式识别为LaTeX 表达式。如针对以下图片 ([docs/examples/math-formula-42.png](docs/examples/math-formula-42.png))：

<div align="center">
  <img src="./docs/examples/math-formula-42.png" alt="Pure Math Formula image" width="300px"/>
</div>


调用方式如下：

```python
from pix2text import Pix2Text

img_fp = './docs/examples/math-formula-42.png'
p2t = Pix2Text()
outs = p2t.recognize_formula(img_fp)
print(outs)
```

返回结果为字符串，即对应的LaTeX 表达式。具体说明见[接口说明](#接口说明)。

### 识别纯文字图片

对于只包含文字不包含数学公式的图片，使用函数 `.recognize_text()` 可以识别出图片中的文字。此时 Pix2Text 相当于一般的文字 OCR 引擎。如针对以下图片 ([docs/examples/general.jpg](docs/examples/general.jpg))：

<div align="center">
  <img src="./docs/examples/general.jpg" alt="Pure Math Formula image" width="400px"/>
</div>


调用方式如下：

```python
from pix2text import Pix2Text

img_fp = './docs/examples/general.jpg'
p2t = Pix2Text()
outs = p2t.recognize_text(img_fp)
print(outs)
```

返回结果为字符串，即对应的文字序列。具体说明见[接口说明](#接口说明)。




## 示例

### 英文

**识别效果**：

![Pix2Text 识别英文](docs/figs/output-en.jpg)

**识别命令**：

```bash
$ p2t predict -l en -a mfd -t yolov7 --analyzer-model-fp ~/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --resized-shape 768 --save-analysis-res out_tmp.jpg --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --auto-line-break -i docs/examples/en1.jpg
```

> 注意 ⚠️ ：上面命令使用了付费版模型，也可以如下使用免费版模型，只是效果略差：
>
> ```bash
> $ p2t predict -l en -a mfd -t yolov7_tiny --resized-shape 768 --save-analysis-res out_tmp.jpg --auto-line-break -i docs/examples/en1.jpg
> ```



### 简体中文

**识别效果**：

![Pix2Text 识别简体中文](docs/figs/output-ch_sim.jpg)

**识别命令**：

```bash
$ p2t predict -l en,ch_sim -a mfd -t yolov7 --analyzer-model-fp ~/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --resized-shape 768 --save-analysis-res out_tmp.jpg --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --auto-line-break -i docs/examples/mixed.jpg
```

> 注意 ⚠️ ：上面命令使用了付费版模型，也可以如下使用免费版模型，只是效果略差：
>
> ```bash
> $ p2t predict -l en,ch_sim -a mfd -t yolov7_tiny --resized-shape 768 --save-analysis-res out_tmp.jpg --auto-line-break -i docs/examples/mixed.jpg
> ```

### 繁体中文

**识别效果**：

![Pix2Text 识别繁体中文](docs/figs/output-ch_tra.jpg)

**识别命令**：

```bash
$ p2t predict -l en,ch_tra -a mfd -t yolov7 --analyzer-model-fp ~/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --resized-shape 768 --save-analysis-res out_tmp.jpg -i docs/examples/ch_tra.jpg
```

> 注意 ⚠️ ：上面命令使用了付费版模型，也可以如下使用免费版模型，只是效果略差：
>
> ```bash
> $ p2t predict -l en,ch_tra -a mfd -t yolov7_tiny --resized-shape 768 --save-analysis-res out_tmp.jpg -i docs/examples/ch_tra.jpg
> ```



### 越南语
**识别效果**：

![Pix2Text 识别越南语](docs/figs/output-vietnamese.jpg)

**识别命令**：

```bash
$ p2t predict -l en,vi -a mfd -t yolov7 --analyzer-model-fp ~/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --resized-shape 768 --save-analysis-res out_tmp.jpg -i docs/examples/vietnamese.jpg
```

> 注意 ⚠️ ：上面命令使用了付费版模型，也可以如下使用免费版模型，只是效果略差：
>
> ```bash
> $ p2t predict -l en,vi -a mfd -t yolov7_tiny --resized-shape 768 --save-analysis-res out_tmp.jpg -i docs/examples/vietnamese.jpg
> ```


## 模型下载

### 开源免费模型

安装好 Pix2Text 后，首次使用时系统会**自动下载** 免费模型文件，并存于 `~/.pix2text/1.0`目录（Windows下默认路径为 `C:\Users\<username>\AppData\Roaming\pix2text\1.0`）。

> **Note**
>
> 如果已成功运行上面的示例，说明模型已完成自动下载，可忽略本节后续内容。



### 付费模型

除了上面免费的开源模型，P2T 也训练了精度更高的数学公式检测和识别模型，这些模型供 **[P2T网页版](https://p2t.breezedeus.com)** 使用，它们的效果也可以在网页版体验。这些模型不是免费的（抱歉开源作者也是要喝咖啡的），具体可参考 [Pix2Text (P2T) | Breezedeus.com](https://www.breezedeus.com/pix2text) 。



## 安装

嗯，顺利的话一行命令即可。

```bash
pip install pix2text
```

如果需要识别**英文**与**简体中文**之外的文字，请使用以下命令安装额外的包：

```bash
pip install pix2text[multilingual]
```

安装速度慢的话，可以指定国内的安装源，如使用阿里云的安装源：

```bash
pip install pix2text -i https://mirrors.aliyun.com/pypi/simple
```



如果是初次使用**OpenCV**，那估计安装都不会很顺利，bless。

**Pix2Text** 主要依赖 [**CnSTD>=1.2.1**](https://github.com/breezedeus/cnstd)、[**CnOCR>=2.2.2.1**](https://github.com/breezedeus/cnocr) ，以及 [**transformers>=4.37.0**](https://github.com/huggingface/transformers) 。如果安装过程遇到问题，也可参考它们的安装说明文档。



> **Warning** 
>
> 如果电脑中从未安装过 `PyTorch`，`OpenCV` python包，初次安装可能会遇到不少问题，但一般都是常见问题，可以自行百度/Google解决。



## 接口说明

### 类初始化

主类为 [**Pix2Text**](pix2text/pix_to_text.py) ，其初始化函数如下：

```python
class Pix2Text(object):

    def __init__(
        self,
        *,
        languages: Union[str, Sequence[str]] = ('en', 'ch_sim'),
        analyzer_config: Dict[str, Any] = None,
        text_config: Dict[str, Any] = None,
        formula_config: Dict[str, Any] = None,
        device: str = None,
        **kwargs,
    ):
```

其中的各参数说明如下：
* `languages` (str or Sequence[str]): 文字识别对应的语言代码序列；默认为 `('en', 'ch_sim')`，表示可识别英文与简体中文；
	
* `analyzer_config` (dict): 分类模型对应的配置信息；默认为 `None`，表示使用默认配置（使用**MFD** Analyzer）：
	
  ```python
  {
        'model_name': 'mfd'  # 可以取值为 'mfd'（MFD），或者 'layout'（版面分析）
  }
	```
	
* `text_config` (dict): 文字识别模型对应的配置信息；默认为 `None`，表示使用默认配置：

  ```python
  {}
  ```

* `formula_config` (dict): 公式识别模型对应的配置信息；默认为 `None`，表示使用默认配置：

  ```python
  {}
  ```
  
* `device` (str): 使用什么资源进行计算，支持 `['cpu', 'cuda', 'gpu', 'mps']` 等；默认为 `None`，表示自动选择设备；

* `**kwargs` (): 预留的其他参数；目前未被使用。



### 识别类函数

#### 识别混合图片

通过调用类 **`Pix2Text`** 的类函数 `.recognize()` 完成对指定图片进行识别。类函数 `.recognize()` 说明如下：

```python
    def recognize(
        self, img: Union[str, Path, Image.Image], **kwargs
    ) -> List[Dict[str, Any]]:
```



其中的输入参数说明如下：

* `img` (`str` or `Image.Image`)：待识别图片的路径，或者利用 `Image.open()` 已读入的图片 `Image` 。
* `kwargs`: 保留字段，可以包含以下值：
  * `resized_shape` (`int`): 把图片宽度resize到此大小再进行处理；默认值为 `700`；
  * `save_analysis_res` (`str`): 把解析结果图片存在此文件中；默认值为 `None`，表示不存储；
  * `embed_sep` (`tuple`): embedding latex的前后缀；只针对使用 `MFD` 时才有效；默认值为 `(' $', '$ ')`；
  * `isolated_sep` (`tuple`): isolated latex的前后缀；只针对使用 `MFD` 时才有效；默认值为 `('$$\n', '\n$$')`；
  * `det_bbox_max_expand_ratio (float)`: 扩展检测到的文本边界框（bbox）的高度。这个值表示相对于原始 bbox 高度的上下最大扩展比例；默认值为 `0.2`。
  * `mfr_batch_size (int)`: MFR 预测时使用的 batch size；默认值为 `1`。



返回结果为列表（`list`），列表中的每个元素为`dict`，包含如下 `key`：

* `type`：识别出的图像类别；
  * 对于 **MFD Analyzer**（数学公式检测），取值为 `text`（纯文本）、`isolated`（独立行的数学公式） 或者 `embedding`（行内的数学公式）；
  * 对于 **Layout Analyzer**（版面分析），取值为版面分析结果类别。
* `text`：识别出的文字或Latex表达式；
* `position`：所在块的位置信息，`np.ndarray`, with shape of `[4, 2]`；
* `line_number`：仅在使用 **MFD Analyzer** 时，才会包含此字段。此字段为 Box 所在的行号（第一行 **`line_number=0`**），值相同的 Box 表示它们在同一行。

  > 注意：此取值从 P2T **v0.2.3** 开始才有，之前版本没有此 `key`。



`Pix2Text` 类也实现了 `__call__()` 函数，其功能与 `.recognize()` 函数完全相同。所以才会有以下的调用方式：

```python
from pix2text import Pix2Text, merge_line_texts

img_fp = './docs/examples/formula.jpg'
p2t = Pix2Text(analyzer_config=dict(model_name='mfd'))
outs = p2t.recognize(img_fp, resized_shape=608)  # 也可以使用 `p2t(img_fp, resized_shape=608)` 获得相同的结果
print(outs)
# 如果只需要识别出的文字和Latex表示，可以使用下面行的代码合并所有结果
only_text = merge_line_texts(outs, auto_line_break=True)
print(only_text)
```



#### 识别纯文字图片

通过调用类 **`Pix2Text`** 的类函数 `.recognize_text()` 完成对指定图片进行文字识别。此时，Pix2Text 提供了一般的文字识别功能。类函数 `.recognize_text()` 说明如下：

```python
    def recognize_text(
        self,
        imgs: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]],
        **kwargs,
    ) -> Union[str, List[str]]:
```



其中的输入参数说明如下：

* `imgs` (`Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]]`)：待识别图片的路径，或者利用 `Image.open()` 已读入的图片 `Image` 。支持单张图片，或者多张图片列表。
* `kwargs`: 传入文字识别接口的其他参数。

返回结果为识别后文本字符串（输入为多张图片时返回是同样长度的列表）。



#### 识别纯公式图片

通过调用类 **`Pix2Text`** 的类函数 `.recognize_formula()` 识别指定图片中的数学公式，并转化为 Latex 表示。类函数 `.recognize_formula()` 说明如下：

```python
    def recognize_formula(
        self,
        imgs: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]],
        batch_size: int = 1,
        **kwargs,
    ) -> Union[str, List[str]]:
```



其中的输入参数说明如下：

* `imgs` (`Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]]`)：待识别图片的路径，或者利用 `Image.open()` 已读入的图片 `Image` 。支持单张图片，或者多张图片列表。
* `batch_size`：batch size。
* `kwargs`: 传入公式识别接口的其他参数。

返回结果为识别后的 Latex 表示字符串（输入为多张图片时返回是同样长度的列表）。






## 脚本使用

**P2T** 包含了以下命令行工具。



### 对单张图片或单个文件夹中的图片进行识别

使用命令 **`p2t predict`** 预测单张图片或文件夹中所有图片，以下是使用说明：

```bash
$ p2t predict -h
用法：p2t predict [选项]

  使用 Pix2Text (P2T) 来预测图像中的文本信息

选项：
  -l, --languages TEXT            文本-OCR识别的语言代码，用逗号分隔
                                  [默认值: en,ch_sim]
  -a, --analyzer-name [mfd|layout]
                                  使用哪种分析器，MFD 或版面分析
                                  [默认值: mfd]
  -t, --analyzer-type TEXT        分析器使用哪种模型，
                                  'yolov7_tiny' 或 'yolov7'  [默认值: yolov7_tiny]
  --analyzer-model-fp TEXT        分析器检测模型的文件路径。
                                  默认值：`无`，表示使用默认模型
  --formula-ocr-config TEXT       LatexOCR数学公式识别模型的配置信息。
                                  默认值：`无`，表示使用默认配置
  --text-ocr-config TEXT          文本-OCR识别的配置信息，以 JSON 字符串格式。
                                  默认值：`无`，表示使用默认配置
  -d, --device TEXT               选择使用 `cpu`、`gpu`，
                                  或特定的 GPU，如 `cuda:0` 运行代码 [默认值: cpu]
--image-type [mixed|formula|text]
                                  处理的图片类型，'mixed'、'formula' 或 'text' [默认值: mixed]
  --resized-shape INTEGER         在处理前将图像宽度调整为此大小 [默认值: 608]
  -i, --img-file-or-dir TEXT      输入图像的文件路径或指定目录  [必需]
  --save-analysis-res TEXT        将分析结果保存到此文件或目录
                                  （如果 '--img-file-or-dir' 是文件/目录，
                                  则 '--save-analysis-res' 也应是文件/目录）。
                                  设为 `无` 表示不保存
  --rec-kwargs TEXT               调用 .recognize() 的 kwargs，以 JSON 字符串格式
  --auto-line-break / --no-auto-line-break
                                  是否自动确定将相邻行结果合并为单行结果
                                  [默认值: no-auto-line-break]
  --log-level TEXT                日志级别，如 `INFO`, `DEBUG`
                                  [默认值: INFO]
  -h, --help                      显示此消息并退出。
```



此命令可用于**打印对指定图片的检测和识别结果**，如运行：

```bash
$ p2t predict -a mfd --resized-shape 608 -i docs/examples/en1.jpg --save-analysis-res output-en1.jpg
```

上面命令打印出识别结果，同时会把检测结果存储在 `output-en1.jpg` 文件中，类似以下效果：


<div align="center">
  <img src="./docs/figs/output-en1.jpg" alt="P2T 数学公式检测效果图" width="600px"/>
</div>


## HTTP服务

 **Pix2Text** 加入了基于 FastAPI 的HTTP服务。开启服务需要安装几个额外的包，可以使用以下命令安装：

```bash
$ pip install pix2text[serve]
```



安装完成后，可以通过以下命令启动HTTP服务（**`-p`** 后面的数字是**端口**，可以根据需要自行调整）：

```bash
$ p2t serve -l en,ch_sim -a mfd
```



`p2t serve` 命令使用说明：

```bash
$ p2t serve -h
用法: p2t serve [OPTIONS]

  开启HTTP服务。

选项：
  -l, --languages TEXT            文本-OCR识别的语言代码，用逗号分隔
                                  [默认值: en,ch_sim]
  -a, --analyzer-name [mfd|layout]
                                  使用哪种分析器，MFD 或版面分析
                                  [默认值: mfd]
  -t, --analyzer-type TEXT        分析器使用哪种模型，
                                  'yolov7_tiny' 或 'yolov7'  [默认值: yolov7_tiny]
  --analyzer-model-fp TEXT        分析器检测模型的文件路径。
                                  默认值：`无`，表示使用默认模型
  --formula-ocr-config TEXT       LatexOCR数学公式识别模型的配置信息。
                                  默认值：`无`，表示使用默认配置
  --text-ocr-config TEXT          文本-OCR识别的配置信息，以 JSON 字符串格式。
                                  默认值：`无`，表示使用默认配置
  -d, --device TEXT               选择使用 `cpu`、`gpu`，
                                  或特定的 GPU，如 `cuda:0` 运行代码 [默认值:
                                  cpu]
  -H, --host TEXT                 服务器主机  [默认值: 0.0.0.0]
  -p, --port INTEGER              服务器端口  [默认值: 8503]
  --reload                        当代码更改时是否重新加载服务器
  --log-level TEXT                日志级别，如 `INFO`, `DEBUG`
                                  [默认值: INFO]
  -h, --help                      显示此消息并退出。
```



服务开启后，可以使用以下方式调用服务。



### Python 调用服务

使用如下方式调用服务，参考文件 [scripts/try_service.py](scripts/try_service.py)：

```python
import requests

url = 'http://0.0.0.0:8503/pix2text'

image_fp = 'docs/examples/mixed.jpg'
data = {
    "image_type": "mixed",  # "mixed": 混合图片；"formula": 纯公式图片；"text": 纯文字图片
    "resized_shape": 768,  # image_type=="mixed" 才有效
    "embed_sep": " $,$ ",  # image_type=="mixed" 才有效
    "isolated_sep": "$$\n, \n$$"  # image_type=="mixed" 才有效
}
files = {
    "image": (image_fp, open(image_fp, 'rb'))
}

r = requests.post(url, data=data, files=files)

outs = r.json()['results']
if isinstance(outs, str):
    only_text = outs
else:
    only_text = '\n'.join([out['text'] for out in outs])
print(f'{only_text=}')
```



### Curl 调用服务

如下使用 `curl` 调用服务：

```bash
$ curl -F image=@docs/examples/mixed.jpg --form 'image_type=mixed' --form 'resized_shape=768' http://0.0.0.0:8503/pix2text
```



### 其他语言调用服务

请参照 `curl` 的调用方式自行实现。




## 给作者来杯咖啡

开源不易，如果此项目对您有帮助，可以考虑 [给作者加点油🥤，鼓鼓气💪🏻](https://www.breezedeus.com/buy-me-coffee) 。

---

官方代码库：[https://github.com/breezedeus/pix2text](https://github.com/breezedeus/pix2text) 。

Pix2Text (P2T) 更多信息：[https://www.breezedeus.com/pix2text](https://www.breezedeus.com/pix2text) 。

