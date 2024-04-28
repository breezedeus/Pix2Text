<figure markdown>

[中文](examples.md) | English

</figure>

# Examples
## Recognize PDF Files and Return Markdown Format

For PDF files, you can use the `.recognize_pdf()` function to recognize the entire file or specific pages and output the results as a Markdown file. For example, for the following PDF file ([examples/test-doc.pdf](examples/test-doc.pdf)):

You can call the function like this:

```python
from pix2text import Pix2Text

img_fp = './examples/test-doc.pdf'
p2t = Pix2Text.from_config()
doc = p2t.recognize_pdf(img_fp, page_numbers=[0, 1])
doc.to_markdown('output-md')  # The exported Markdown information is saved in the output-md directory
```

You can also achieve the same functionality using the command line. Below is a command that uses the premium models (MFD + MFR + CnOCR) for recognition:

```bash
p2t predict -l en,ch_sim --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --rec-kwargs '{"page_numbers": [0, 1]}' --resized-shape 768 --file-type pdf -i docs/examples/test-doc.pdf -o output-md --save-debug-res output-debug
```

You can find the recognition result in [output-md/output.md](output-md/output.md).

<br/>

> If you wish to export formats other than Markdown, such as Word, HTML, PDF, etc., it is recommended to use the tool [Pandoc](https://pandoc.org) to convert the Markdown result.

## Recognize Images with Complex Layout

You can use the `.recognize_page()` function to recognize text and mathematical formulas in images. For example, for the following image ([examples/page2.png](examples/page2.png)):

<div align="center">
  <img src="https://pix2text.readthedocs.io/zh/latest/examples/page2.png" alt="Page-image" width="600px"/>
</div>

You can call the function like this:

```python
from pix2text import Pix2Text

img_fp = './examples/test-doc.pdf'
p2t = Pix2Text.from_config()
page = p2t.recognize_page(img_fp)
page.to_markdown('output-page')  # The exported Markdown information is saved in the output-page directory
```

You can also achieve the same functionality using the command line. Below is a command that uses the premium models (MFD + MFR + CnOCR) for recognition:

```bash
p2t predict -l en,ch_sim --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --resized-shape 768 --file-type page -i docs/examples/page2.png -o output-page --save-debug-res output-debug-page
```

The recognition result is similar to [output-md/output.md](output-md/output.md).


## Recognize Paragraph Images with Both Formulas and Texts

For paragraph images containing both formulas and texts, you don't need to use the layout analysis model. You can use the `.recognize_text_formula()` function to recognize both texts and mathematical formulas in the image. For example, for the following image ([examples/en1.jpg](examples/en1.jpg)):

<div align="center">
  <img src="https://pix2text.readthedocs.io/zh/latest/examples/en1.jpg" alt="English-mixed-image" width="600px"/>
</div>

You can call the function like this:

```python
from pix2text import Pix2Text, merge_line_texts

img_fp = './examples/en1.jpg'
p2t = Pix2Text.from_config()
outs = p2t.recognize_text_formula(img_fp, resized_shape=768, return_text=True)
print(outs)
```

The returned result `outs` is a dictionary, where the key `position` represents the box position information, `type` represents the category information, and `text` represents the recognition result. For detailed explanations, see [API Documentation](#api-documentation).

You can also achieve the same functionality using the command line. Below is a command that uses the premium models (MFD + MFR + CnOCR) for recognition:

```bash
p2t predict -l en,ch_sim --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --resized-shape 768 --file-type text_formula -i docs/examples/en1.jpg
```

Or use the free open-source models for recognition:

```bash
p2t predict -l en,ch_sim --resized-shape 768 --file-type text_formula -i docs/examples/en1.jpg
```

## Recognize Pure Formula Images

For images containing only mathematical formulas, you can use the `.recognize_formula()` function to recognize the formulas as LaTeX expressions. For example, for the following image ([examples/math-formula-42.png](examples/math-formula-42.png)):

<div align="center">
  <img src="https://pix2text.readthedocs.io/zh/latest/examples/math-formula-42.png" alt="Pure-Math-Formula-image" width="300px"/>
</div>

You can call the function like this:

```python
from pix2text import Pix2Text

img_fp = './examples/math-formula-42.png'
p2t = Pix2Text.from_config()
outs = p2t.recognize_formula(img_fp)
print(outs)
```

The returned result is a string representing the corresponding LaTeX expression. For detailed explanations, see [Usage](usage.md).

You can also achieve the same functionality using the command line. Below is a command that uses the premium model (MFR) for recognition:

```bash
p2t predict -l en,ch_sim --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --file-type formula -i docs/examples/math-formula-42.png
```

Or use the free open-source model for recognition:

```bash
p2t predict -l en,ch_sim --file-type textformula -i docs/examples/math-formula-42.png
```

## Recognize Pure Text Images

For images containing only text without mathematical formulas, you can use the `.recognize_text()` function to recognize the text in the image. In this case, Pix2Text acts as a general text OCR engine. For example, for the following image ([examples/general.jpg](examples/general.jpg)):

<div align="center">
  <img src="https://pix2text.readthedocs.io/zh/latest/examples/general.jpg" alt="Pure-Math-Formula-image" width="400px"/>
</div>

You can call the function like this:

```python
from pix2text import Pix2Text

img_fp = './examples/general.jpg'
p2t = Pix2Text.from_config()
outs = p2t.recognize_text(img_fp)
print(outs)
```

The returned result is a string representing the corresponding text sequence. For detailed explanations, see [API Documentation](https://pix2text.readthedocs.io/zh/latest/pix2text/pix_to_text/).

You can also achieve the same functionality using the command line. Below is a command that uses the premium model (CnOCR) for recognition:

```bash
p2t predict -l en,ch_sim --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --file-type text -i docs/examples/general.jpg
```

Or use the free open-source model for recognition:

```bash
p2t predict -l en,ch_sim --file-type text -i docs/examples/general.jpg
```

## For Different Languages

### English

**Recognition Result**:

![Pix2Text Recognizing English](figs/output-en.jpg)

**Recognition Command**:

```bash
p2t predict -l en --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --resized-shape 768 --file-type text_formula -i docs/examples/en1.jpg
```

### Simplified Chinese

**Recognition Result**:

![Pix2Text Recognizing Simplified Chinese](figs/output-ch_sim.jpg)

**Recognition Command**:

```bash
p2t predict -l en,ch_sim --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --resized-shape 768 --auto-line-break --file-type text_formula -i docs/examples/mixed.jpg
```

### Traditional Chinese

**Recognition Result**:

![Pix2Text Recognizing Traditional Chinese](figs/output-ch_tra.jpg)

**Recognition Command**:

```bash
p2t predict -l en,ch_tra --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --resized-shape 768 --auto-line-break --file-type text_formula -i docs/examples/ch_tra.jpg
```

> Note ⚠️: Please install the multilingual version of pix2text using the following command:
> ```bash
> pip install pix2text[multilingual]
> ```

### Vietnamese

**Recognition Result**:

![Pix2Text Recognizing Vietnamese](figs/output-vietnamese.jpg)

**Recognition Command**:

```bash
p2t predict -l en,vi --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --resized-shape 768 --no-auto-line-break --file-type text_formula -i docs/examples/vietnamese.jpg
```

> Note ⚠️: Please install the multilingual version of pix2text using the following command:
> ```bash
> pip install pix2text[multilingual]
> ```