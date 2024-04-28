<figure markdown>

[English](examples_en.md) | 中文

</figure>

# 示例
## 识别 PDF 文件，返回其 Markdown 格式

对于 PDF 文件，可以使用函数 `.recognize_pdf()` 对整个文件或者指定页进行识别，并把结果输出为 Markdown 文件。如针对以下 PDF 文件 ([examples/test-doc.pdf](examples/test-doc.pdf))：

调用方式如下：

```python
from pix2text import Pix2Text

img_fp = './examples/test-doc.pdf'
p2t = Pix2Text.from_config()
doc = p2t.recognize_pdf(img_fp, page_numbers=[0, 1])
doc.to_markdown('output-md')  # 导出的 Markdown 信息保存在 output-md 目录中
```

也可以使用命令行完成一样的功能，如下面命令使用了付费版模型（MFD + MFR + CnOCR 三个付费模型）进行识别：

```bash
p2t predict -l en,ch_sim --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --rec-kwargs '{"page_numbers": [0, 1]}' --resized-shape 768 --file-type pdf -i docs/examples/test-doc.pdf -o output-md --save-debug-res output-debug
```

识别结果见 [output-md/output.md](output-md/output.md)。

<br/>

> 如果期望导出 Markdown 之外的其他格式，如 Word、HTML、PDF 等，推荐使用工具 [Pandoc](https://pandoc.org) 对 Markdown 结果进行转换即可。

## 识别带有复杂排版的图片
可以使用函数 `.recognize_page()` 识别图片中的文字和数学公式。如针对以下图片 ([examples/page2.png](examples/page2.png))：

<div align="center">
  <img src="https://pix2text.readthedocs.io/zh/latest/examples/page2.png" alt="Page-image" width="600px"/>
</div>

调用方式如下：

```python
from pix2text import Pix2Text

img_fp = './examples/test-doc.pdf'
p2t = Pix2Text.from_config()
page = p2t.recognize_page(img_fp)
page.to_markdown('output-page')  # 导出的 Markdown 信息保存在 output-page 目录中
```

也可以使用命令行完成一样的功能，如下面命令使用了付费版模型（MFD + MFR + CnOCR 三个付费模型）进行识别：

```bash
p2t predict -l en,ch_sim --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --resized-shape 768 --file-type page -i docs/examples/page2.png -o output-page --save-debug-res output-debug-page
```

识别结果和 [output-md/output.md](output-md/output.md) 类似。

## 识别既有公式又有文本的段落图片

对于既有公式又有文本的段落图片，识别时不需要使用版面分析模型。
可以使用函数 `.recognize_text_formula()` 识别图片中的文字和数学公式。如针对以下图片 ([examples/en1.jpg](examples/en1.jpg))：

<div align="center">
  <img src="https://pix2text.readthedocs.io/zh/latest/examples/en1.jpg" alt="English-mixed-image" width="600px"/>
</div>

调用方式如下：

```python
from pix2text import Pix2Text, merge_line_texts

img_fp = './examples/en1.jpg'
p2t = Pix2Text.from_config()
outs = p2t.recognize_text_formula(img_fp, resized_shape=768, return_text=True)
print(outs)
```

返回结果 `outs` 是个 `dict`，其中 key `position` 表示Box位置信息，`type` 表示类别信息，而 `text` 表示识别的结果。具体说明见[接口说明](#接口说明)。

也可以使用命令行完成一样的功能，如下面命令使用了付费版模型（MFD + MFR + CnOCR 三个付费模型）进行识别：

```bash
p2t predict -l en,ch_sim --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --resized-shape 768 --file-type text_formula -i docs/examples/en1.jpg
```

或者使用免费开源模型进行识别：

```bash
p2t predict -l en,ch_sim --resized-shape 768 --file-type text_formula -i docs/examples/en1.jpg
```

## 识别纯公式图片

对于只包含数学公式的图片，使用函数 `.recognize_formula()` 可以把数学公式识别为 LaTeX 表达式。如针对以下图片 ([examples/math-formula-42.png](examples/math-formula-42.png))：

<div align="center">
  <img src="https://pix2text.readthedocs.io/zh/latest/examples/math-formula-42.png" alt="Pure-Math-Formula-image" width="300px"/>
</div>


调用方式如下：

```python
from pix2text import Pix2Text

img_fp = './examples/math-formula-42.png'
p2t = Pix2Text.from_config()
outs = p2t.recognize_formula(img_fp)
print(outs)
```

返回结果为字符串，即对应的 LaTeX 表达式。具体说明见[说明](usage.md)。

也可以使用命令行完成一样的功能，如下面命令使用了付费版模型（MFR 一个付费模型）进行识别：

```bash
p2t predict -l en,ch_sim --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --file-type formula -i docs/examples/math-formula-42.png
```

或者使用免费开源模型进行识别：

```bash
p2t predict -l en,ch_sim --file-type textformula -i docs/examples/math-formula-42.png
```

## 识别纯文字图片

对于只包含文字不包含数学公式的图片，使用函数 `.recognize_text()` 可以识别出图片中的文字。此时 Pix2Text 相当于一般的文字 OCR 引擎。如针对以下图片 ([examples/general.jpg](examples/general.jpg))：

<div align="center">
  <img src="https://pix2text.readthedocs.io/zh/latest/examples/general.jpg" alt="Pure-Math-Formula-image" width="400px"/>
</div>


调用方式如下：

```python
from pix2text import Pix2Text

img_fp = './examples/general.jpg'
p2t = Pix2Text.from_config()
outs = p2t.recognize_text(img_fp)
print(outs)
```

返回结果为字符串，即对应的文字序列。具体说明见[接口说明](https://pix2text.readthedocs.io/zh/latest/pix2text/pix_to_text/)。

也可以使用命令行完成一样的功能，如下面命令使用了付费版模型（CnOCR 一个付费模型）进行识别：

```bash
p2t predict -l en,ch_sim --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --file-type text -i docs/examples/general.jpg
```

或者使用免费开源模型进行识别：

```bash
p2t predict -l en,ch_sim --file-type text -i docs/examples/general.jpg
```


## 针对不同语言

### 英文

**识别效果**：

![Pix2Text 识别英文](figs/output-en.jpg)

**识别命令**：

```bash
p2t predict -l en --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --resized-shape 768 --file-type text_formula -i docs/examples/en1.jpg
```

### 简体中文

**识别效果**：

![Pix2Text 识别简体中文](figs/output-ch_sim.jpg)

**识别命令**：

```bash
p2t predict -l en,ch_sim --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --resized-shape 768 --auto-line-break --file-type text_formula -i docs/examples/mixed.jpg
```

### 繁体中文

**识别效果**：

![Pix2Text 识别繁体中文](figs/output-ch_tra.jpg)

**识别命令**：

```bash
p2t predict -l en,ch_tra --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --resized-shape 768 --auto-line-break --file-type text_formula -i docs/examples/ch_tra.jpg
```

> 注意 ⚠️ ：请通过以下命令安装 pix2text 的多语言版本：
> ```bash
> pip install pix2text[multilingual]
> ```


### 越南语
**识别效果**：

![Pix2Text 识别越南语](figs/output-vietnamese.jpg)

**识别命令**：

```bash
p2t predict -l en,vi --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --resized-shape 768 --no-auto-line-break --file-type text_formula -i docs/examples/vietnamese.jpg
```

> 注意 ⚠️ ：请通过以下命令安装 pix2text 的多语言版本：
> ```bash
> pip install pix2text[multilingual]
> ```
