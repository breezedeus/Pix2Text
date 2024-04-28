# Usage

## 模型文件自动下载

首次使用 **Pix2Text** 时，系统会**自动下载**所需的开源模型，并存于 `~/.pix2text` 目录（Windows下默认路径为 `C:\Users\<username>\AppData\Roaming\pix2text`）。
CnOCR 和 CnSTD 中的模型分别存于 `~/.cnocr` 和 `~/.cnstd` 中（Windows 下默认路径为 `C:\Users\<username>\AppData\Roaming\cnocr` 和 `C:\Users\<username>\AppData\Roaming\cnstd`）。
下载过程请耐心等待，无法科学上网时系统会自动尝试其他可用站点进行下载，所以可能需要等待较长时间。
对于没有网络连接的机器，可以先把模型下载到其他机器上，然后拷贝到对应目录。

如果系统无法自动成功下载模型文件，则需要手动下载模型文件，可以参考 [huggingface.co/breezedeus](https://huggingface.co/breezedeus) （[国内链接](https://hf-mirror.com/breezedeus)）自己手动下载。
具体说明见 [模型下载](models.md)。


## 初始化
### 方法一

类 [Pix2Text](pix2text/pix_to_text.md) 是识别主类，包含了多个识别函数识别不同类型的 **图片** 或 **PDF文件** 中的内容。类 `Pix2Text` 的初始化函数如下：

```python
class Pix2Text(object):
    def __init__(
        self,
        *,
        layout_parser: Optional[LayoutParser] = None,
        text_formula_ocr: Optional[TextFormulaOCR] = None,
        table_ocr: Optional[TableOCR] = None,
        **kwargs,
    ):
		"""
        Initialize the Pix2Text object.
        Args:
            layout_parser (LayoutParser): The layout parser object; default value is `None`, which means to create a default one
            text_formula_ocr (TextFormulaOCR): The text and formula OCR object; default value is `None`, which means to create a default one
            table_ocr (TableOCR): The table OCR object; default value is `None`, which means not to recognize tables
            **kwargs (dict): Other arguments, currently not used
        """
```

其中的几个参数含义如下：

* `layout_parser`：版面分析模型对象，默认值为 `None`，表示使用默认的版面分析模型；
* `text_formula_ocr`：文字与公式识别模型对象，默认值为 `None`，表示使用默认的文字与公式识别模型；
* `table_ocr`：表格识别模型对象，默认值为 `None`，表示不识别表格；
* `**kwargs`：其他参数，目前未使用。


每个参数都有默认取值，所以可以不传入任何参数值进行初始化：`p2t = Pix2Text()`。但请注意，如果不传入任何参数值，那么只会导入默认的版面分析模型和文字与公式识别模型，而**不会导入表格识别模型**。

初始化 Pix2Text 实例的更好的方法是使用以下的函数。

### 方法二
可以通过指定配置信息来初始化 `Pix2Text` 类的实例：

```python
@classmethod
def from_config(
		cls,
		total_configs: Optional[dict] = None,
		enable_formula: bool = True,
		enable_table: bool = True,
		device: str = None,
		**kwargs,
):
	"""
    Create a Pix2Text object from the configuration.
    Args:
        total_configs (dict): The total configuration; default value is `None`, which means to use the default configuration.
            If not None, it should contain the following keys:

                * `layout`: The layout parser configuration
                * `text_formula`: The TextFormulaOCR configuration
                * `table`: The table OCR configuration
        enable_formula (bool): Whether to enable formula recognition; default value is `True`
        enable_table (bool): Whether to enable table recognition; default value is `True`
        device (str): The device to run the model; optional values are 'cpu', 'gpu' or 'cuda';
            default value is `None`, which means to select the device automatically
        **kwargs (dict): Other arguments

    Returns: a Pix2Text object

    """
```

其中的几个参数含义如下：

* `total_configs`：总配置，包含以下几个键值：
	- `layout`：版面分析模型的配置；
	- `text_formula`：文字与公式识别模型的配置；
	- `table`：表格识别模型的配置；
  默认值为 `None`，表示使用默认配置。
* `enable_formula`：是否启用公式识别，默认值为 `True`；
* `enable_table`：是否启用表格识别，默认值为 `True`；
* `device`：运行模型的设备，可选值为 `'cpu'`, `'gpu'` 或 `'cuda'`，默认值为 `None`，表示自动选择设备；
* `**kwargs`：其他参数，目前未使用。

这个函数的返回值是一个 `Pix2Text` 类的实例，可以直接使用这个实例进行识别。

推荐使用此函数初始化 Pix2Text 的实例，如：`p2t = Pix2Text.from_config()`。

一个包含配置信息的示例如下：

```python
import os
from pix2text import Pix2Text

text_formula_config = dict(
	languages=('en', 'ch_sim'),  # 设置识别的语言
	mfd=dict(  # 声明 LayoutAnalyzer 的初始化参数
		model_type='yolov7',  # 表示使用的是 YoloV7 模型，而不是 YoloV7_Tiny 模型
		model_fp=os.path.expanduser(
			'~/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt'
		),  # 注：修改成你的模型文件所存储的路径
	),
	formula=dict(
		model_name='mfr-pro',
		model_backend='onnx',
		model_dir=os.path.expanduser(
			'~/.pix2text/1.0/mfr-pro-onnx'
		),  # 注：修改成你的模型文件所存储的路径
	),
	text=dict(
		rec_model_name='doc-densenet_lite_666-gru_large',
		rec_model_backend='onnx',
		rec_model_fp=os.path.expanduser(
			'~/.cnocr/2.3/doc-densenet_lite_666-gru_large/cnocr-v2.3-doc-densenet_lite_666-gru_large-epoch=005-ft-model.onnx'
			# noqa
		),  # 注：修改成你的模型文件所存储的路径
	),
)
total_config = {
	'layout': {'scores_thresh': 0.45},
	'text_formula': text_formula_config,
}
p2t = Pix2Text.from_config(total_configs=total_config)
```

更多初始化的示例请参见 [tests/test_pix2text.py](https://github.com/breezedeus/Pix2Text/blob/main/tests/test_pix2text.py)。

## 各种识别接口

类 `Pix2Text` 提供了不同的识别函数来识别不同类似的图片或者 PDF 文件内容，下面分别说明。


### 1. 函数 `.recognize_pdf()`

此函数用于识别一整个 PDF 文件中的内容。**PDF 文件的内容可以只包含图片而无文字内容**，
如示例文件 [examples/test-doc.pdf](examples/test-doc.pdf)。
识别时，可以指定识别的页数，也可以指定识别的 PDF 文件编号。
函数定义如下：

```python
def recognize_pdf(
		self,
		pdf_fp: Union[str, Path],
		pdf_number: int = 0,
		pdf_id: Optional[str] = None,
		page_numbers: Optional[List[int]] = None,
		**kwargs,
) -> Document:
	"""
    recognize a pdf file
    Args:
        pdf_fp (Union[str, Path]): pdf file path
        pdf_number (int): pdf number
        pdf_id (str): pdf id
        page_numbers (List[int]): page numbers to recognize; default is `None`, which means to recognize all pages
        kwargs (dict): Optional keyword arguments. The same as `recognize_page`

    Returns: a Document object. Use `doc.to_markdown('output-dir')` to get the markdown output of the recognized document.

    """
```

**函数说明**：

* 输入参数 `pdf_fp`：PDF 文件的路径；
* 输入参数 `pdf_number`：PDF 文件的编号，默认值为 `0`；
* 输入参数 `pdf_id`：PDF 文件的 ID，默认值为 `None`；
* 输入参数 `page_numbers`：需要识别的页码列表（页码从 0 开始计数，如 `[0, 1]` 表示只识别文件的第 1、2 页内容），默认值为 `None`，表示识别所有页；
* 输入参数 `**kwargs`：其他参数，具体说明参考下面的函数 `recognize_page()`。

**返回值**：返回一个 `Document` 对象，可以使用 `doc.to_markdown('output-dir')` 来获取识别结果的 markdown 输出。

**调用示例**：

```python
from pix2text import Pix2Text

img_fp = 'examples/test-doc.pdf'
p2t = Pix2Text.from_config()
out_md = p2t.recognize_pdf(
	img_fp,
	page_numbers=[0, 1],
	table_as_image=True,
	save_debug_res=f'./output-debug',
)
out_md.to_markdown('output-pdf-md')
```

### 2. 函数 `.recognize_page()`

此函数用于识别一张包含复杂排版的页面图片中的内容。图片可以包含多列、图片、表格等内容，如示例图片 [examples/page2.png](examples/page2.png)。
函数定义如下：

```python
def recognize_page(
		self,
		img: Union[str, Path, Image.Image],
		page_number: int = 0,
		page_id: Optional[str] = None,
		**kwargs,
) -> Page:
	"""
    Analyze the layout of the image, and then recognize the information contained in each section.

    Args:
        img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
        page_number (str): page number; default value is `0`
        page_id (str): page id; default value is `None`, which means to use the `str(page_number)`
        kwargs ():
            * resized_shape (int): Resize the image width to this size for processing; default value is `768`
            * mfr_batch_size (int): batch size for MFR; When running on GPU, this value is suggested to be set to greater than 1; default value is `1`
            * embed_sep (tuple): Prefix and suffix for embedding latex; only effective when `return_text` is `True`; default value is `(' $', '$ ')`
            * isolated_sep (tuple): Prefix and suffix for isolated latex; only effective when `return_text` is `True`; default value is two-dollar signs
            * line_sep (str): The separator between lines of text; only effective when `return_text` is `True`; default value is a line break
            * auto_line_break (bool): Automatically line break the recognized text; only effective when `return_text` is `True`; default value is `True`
            * det_text_bbox_max_width_expand_ratio (float): Expand the width of the detected text bbox. This value represents the maximum expansion ratio above and below relative to the original bbox height; default value is `0.3`
            * det_text_bbox_max_height_expand_ratio (float): Expand the height of the detected text bbox. This value represents the maximum expansion ratio above and below relative to the original bbox height; default value is `0.2`
            * embed_ratio_threshold (float): The overlap threshold for embed formulas and text lines; default value is `0.6`.
                When the overlap between an embed formula and a text line is greater than or equal to this threshold,
                the embed formula and the text line are considered to be on the same line;
                otherwise, they are considered to be on different lines.
            * table_as_image (bool): If `True`, the table will be recognized as an image (don't parse the table content as text) ; default value is `False`
            * title_contain_formula (bool): If `True`, the title of the page will be recognized as a mixed image (text and formula). If `False`, it will be recognized as a text; default value is `False`
            * text_contain_formula (bool): If `True`, the text of the page will be recognized as a mixed image (text and formula). If `False`, it will be recognized as a text; default value is `True`
            * formula_rec_kwargs (dict): generation arguments passed to formula recognizer `latex_ocr`; default value is `{}`
            * save_debug_res (str): if `save_debug_res` is set, the directory to save the debug results; default value is `None`, which means not to save

    Returns: a Page object. Use `page.to_markdown('output-dir')` to get the markdown output of the recognized page.
    """
```

**函数说明**：

* 输入参数 `img`：图片路径或者 `Image.Image` 对象；
* 输入参数 `page_number`：页码，默认值为 `0`；
* 输入参数 `page_id`：页码 ID，默认值为 `None`，此时会使用 `str(page_number)` 作为其取值；
* kwargs：其他参数，具体说明如下：
	- `resized_shape`：调整图片的宽度为此大小以进行处理，默认值为 `768`；
	- `mfr_batch_size`：MFR 预测时使用的批大小；在 GPU 上运行时，建议将此值设置为大于 `1`；默认值为 `1`；
	- `embed_sep`：嵌入 LaTeX 的前缀和后缀；仅在 `return_text` 为 `True` 时有效；默认值为 `(' $', '$ ')`；
	- `isolated_sep`：孤立 LaTeX 的前缀和后缀；仅在 `return_text` 为 `True` 时有效；默认值为两个美元符号；
	- `line_sep`：文本行之间的分隔符；仅在 `return_text` 为 `True` 时有效；默认值为换行符；
	- `auto_line_break`：自动换行识别的文本；仅在 `return_text` 为 `True` 时有效；默认值为 `True`；
	- `det_text_bbox_max_width_expand_ratio`：扩展检测文本框的宽度。此值表示相对于原始框高度的最大扩展比率；默认值为 `0.3`；
	- `det_text_bbox_max_height_expand_ratio`：扩展检测文本框的高度。此值表示相对于原始框高度的最大扩展比率；默认值为 `0.2`；
	- `embed_ratio_threshold`：嵌入公式和文本行之间的重叠阈值；默认值为 `0.6`。当嵌入公式和文本行之间的重叠大于或等于此阈值时，认为嵌入公式和文本行在同一行；否则，认为它们在不同行
    - `table_as_image`：如果为 `True`，则将表格识别为图像（不将表格内容解析为文本）；默认值为 `False`
    - `title_contain_formula`：如果为 `True`，则将页面标题作为为混合图像（文本和公式）进行识别。如果为 `False`，则将其作为文本图片进行识别（不识别公式）；默认值为 `False`
    - `text_contain_formula`：如果为 `True`，则将页面文本作为混合图像（文本和公式）进行识别。如果为 `False`，则将其作为文本进行识别（不识别公式）；默认值为 `True`
    - `formula_rec_kwargs`：传递给公式识别器 `latex_ocr` 的生成参数；默认值为 `{}`
    - `save_debug_res`：如果设置了 `save_debug_res`，则把各种中间的解析结果存入此目录以便于调试；默认值为 `None`，表示不保存

**返回值**：返回一个 `Page` 对象，可以使用 `page.to_markdown('output-dir')` 来获取识别结果的 markdown 输出。

**调用示例**：

```python
from pix2text import Pix2Text

img_fp = 'examples/page2.png'
p2t = Pix2Text.from_config()
out_page = p2t.recognize_page(
	img_fp,
	title_contain_formula=False,
	text_contain_formula=False,
	save_debug_res=f'./output-debug',
)
out_page.to_markdown('output-page-md')
```


### 3. 函数 `.recognize_text_formula()`

此函数用于识别一张包含文字和公式的图片（如段落截图）中的内容，如示例图片 [examples/mixed.jpg](examples/mixed.jpg)。
函数定义如下：

```python
def recognize_text_formula(
		self, img: Union[str, Path, Image.Image], return_text: bool = True, **kwargs,
) -> Union[str, List[str], List[Any], List[List[Any]]]:
	"""
    Analyze the layout of the image, and then recognize the information contained in each section.

    Args:
        img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
        return_text (bool): Whether to return the recognized text; default value is `True`
        kwargs ():
            * resized_shape (int): Resize the image width to this size for processing; default value is `768`
            * save_analysis_res (str): Save the mfd result image in this file; default is `None`, which means not to save
            * mfr_batch_size (int): batch size for MFR; When running on GPU, this value is suggested to be set to greater than 1; default value is `1`
            * embed_sep (tuple): Prefix and suffix for embedding latex; only effective when `return_text` is `True`; default value is `(' $', '$ ')`
            * isolated_sep (tuple): Prefix and suffix for isolated latex; only effective when `return_text` is `True`; default value is two-dollar signs
            * line_sep (str): The separator between lines of text; only effective when `return_text` is `True`; default value is a line break
            * auto_line_break (bool): Automatically line break the recognized text; only effective when `return_text` is `True`; default value is `True`
            * det_text_bbox_max_width_expand_ratio (float): Expand the width of the detected text bbox. This value represents the maximum expansion ratio above and below relative to the original bbox height; default value is `0.3`
            * det_text_bbox_max_height_expand_ratio (float): Expand the height of the detected text bbox. This value represents the maximum expansion ratio above and below relative to the original bbox height; default value is `0.2`
            * embed_ratio_threshold (float): The overlap threshold for embed formulas and text lines; default value is `0.6`.
                When the overlap between an embed formula and a text line is greater than or equal to this threshold,
                the embed formula and the text line are considered to be on the same line;
                otherwise, they are considered to be on different lines.
            * table_as_image (bool): If `True`, the table will be recognized as an image; default value is `False`
            * formula_rec_kwargs (dict): generation arguments passed to formula recognizer `latex_ocr`; default value is `{}`

    Returns: a str when `return_text` is `True`; or a list of ordered (top to bottom, left to right) dicts when `return_text` is `False`,
        with each dict representing one detected box, containing keys:

           * `type`: The category of the image; Optional: 'text', 'isolated', 'embedding'
           * `text`: The recognized text or Latex formula
           * `score`: The confidence score [0, 1]; the higher, the more confident
           * `position`: Position information of the block, `np.ndarray`, with shape of [4, 2]
           * `line_number`: The line number of the box (first line `line_number==0`), boxes with the same value indicate they are on the same line

    """
```

**函数说明**：

* 输入参数 `img`：图片路径或者 `Image.Image` 对象；
* 输入参数 `return_text`：是否返回纯文本；取值为 `False` 时返回带有结构化信息的 list；默认值为 `True`；
* 输入参数 `kwargs`：其他参数，具体说明如下：
	- `resized_shape`：调整图片的宽度为此大小以进行处理，默认值为 `768`；
	- `save_analysis_res`：保存 MFD 解析结果图像的文件名；默认值为 `None`，表示不保存；
	- `mfr_batch_size`：MFR 预测时使用的批大小；在 GPU 上运行时，建议将此值设置为大于 `1`；默认值为 `1`；
	- `embed_sep`：嵌入 LaTeX 的前缀和后缀；仅在 `return_text` 为 `True` 时有效；默认值为 `(' $', '$ ')`；
	- `isolated_sep`：孤立 LaTeX 的前缀和后缀；仅在 `return_text` 为 `True` 时有效；默认值为两个美元符号；
	- `line_sep`：文本行之间的分隔符；仅在 `return_text` 为 `True` 时有效；默认值为换行符；
	- `auto_line_break`：自动换行识别的文本；仅在 `return_text` 为 `True` 时有效；默认值为 `True`；
	- `det_text_bbox_max_width_expand_ratio`：扩展检测文本框的宽度。此值表示相对于原始框高度的最大扩展比率；默认值为 `0.3`；
	- `det_text_bbox_max_height_expand_ratio`：扩展检测文本框的高度。此值表示相对于原始框高度的最大扩展比率；默认值为 `0.2`；
	- `embed_ratio_threshold`：嵌入公式和文本行之间的重叠阈值；默认值为 `0.6`。当嵌入公式和文本行之间的重叠大于或等于此阈值时，认为嵌入公式和文本行在同一行；否则，认
    - `table_as_image`：如果为 `True`，则将表格识别为图像；默认值为 `False`
    - `formula_rec_kwargs`：传递给公式识别器 `latex_ocr` 的生成参数；默认值为 `{}`

**返回值**：当 `return_text` 为 `True` 时，返回一个字符串；当 `return_text` 为 `False` 时，返回一个有序的（从上到下，从左到右）字典列表，每个字典表示一个检测框，包含以下键值：
	- `type`：图像的类别；可选值：'text'、'isolated'、'embedding'
	- `text`：识别的文本或 LaTeX 公式
	- `score`：置信度分数 [0, 1]；分数越高，置信度越高
	- `position`：块的位置信息，`np.ndarray`，形状为 `[4, 2]`
	- `line_number`：框的行号（第一行 `line_number==0`），具有相同值的框表示它们在同一行

**调用示例**：

```python
from pix2text import Pix2Text

img_fp = 'examples/mixed.jpg'
p2t = Pix2Text.from_config()
out = p2t.recognize_text_formula(
	img_fp,
	save_analysis_res=f'./output-debug',
)
```

### 4. 函数 `.recognize_formula()`

此函数用于识别一张纯公式的图片中的内容，如示例图片 [examples/formula2.png](examples/formula2.png)。
函数定义如下：

```python
def recognize_formula(
		self,
		imgs: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]],
		batch_size: int = 1,
		return_text: bool = True,
		rec_config: Optional[dict] = None,
		**kwargs,
) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
	"""
    Recognize pure Math Formula images to LaTeX Expressions
    Args:
        imgs (Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]): The image or list of images
        batch_size (int): The batch size
        return_text (bool): Whether to return only the recognized text; default value is `True`
        rec_config (Optional[dict]): The config for recognition
        **kwargs (): Special model parameters. Not used for now

    Returns: The LaTeX Expression or list of LaTeX Expressions;
        str or List[str] when `return_text` is True;
        Dict[str, Any] or List[Dict[str, Any]] when `return_text` is False, with the following keys:

            * `text`: The recognized LaTeX text
            * `score`: The confidence score [0, 1]; the higher, the more confident

    """
```

**函数说明**：

* 输入参数 `imgs`：图片路径或者 `Image.Image` 对象，或者图片路径或者 `Image.Image` 对象的列表；
* 输入参数 `batch_size`：批大小，默认值为 `1`；
* 输入参数 `return_text`：是否返回纯文本；取值为 `False` 时返回带有结构化信息的 list；默认值为 `True`；
* 输入参数 `rec_config`：识别配置，可选值；
* 输入参数 `kwargs`：其他参数，目前未使用。

**返回值**：当 `return_text` 为 `True` 时，返回一个字符串；当 `return_text` 为 `False` 时，返回一个有序的（从上到下，从左到右）字典列表，每个字典表示一个检测框，包含以下键值：
	- `text`：识别的 LaTeX 文本
	- `score`：置信度分数 [0, 1]；分数越高，置信度越高

**调用示例**：

```python
from pix2text import Pix2Text

img_fp = 'examples/formula2.png'
p2t = Pix2Text.from_config()
out = p2t.recognize_formula(
	img_fp,
	save_analysis_res=f'./output-debug',
)
```

### 5. 函数 `.recognize_text()`

此函数用于识别一张纯文字的图片中的内容，如示例图片 [examples/general.jpg](examples/general.jpg)。
函数定义如下：

```python
def recognize_text(
		self,
		imgs: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]],
		return_text: bool = True,
		rec_config: Optional[dict] = None,
		**kwargs,
) -> Union[str, List[str], List[Any], List[List[Any]]]:
	"""
    Recognize a pure Text Image.
    Args:
        imgs (Union[str, Path, Image.Image], List[str], List[Path], List[Image.Image]): The image or list of images
        return_text (bool): Whether to return only the recognized text; default value is `True`
        rec_config (Optional[dict]): The config for recognition
        kwargs (): Other parameters for `text_ocr.ocr()`

    Returns: Text str or list of text strs when `return_text` is True;
        `List[Any]` or `List[List[Any]]` when `return_text` is False, with the same length as `imgs` and the following keys:

            * `position`: Position information of the block, `np.ndarray`, with a shape of [4, 2]
            * `text`: The recognized text
            * `score`: The confidence score [0, 1]; the higher, the more confident

    """
```

**函数说明**：

* 输入参数 `imgs`：图片路径或者 `Image.Image` 对象，或者图片路径或者 `Image.Image` 对象的列表；
* 输入参数 `return_text`：是否返回纯文本；取值为 `False` 时返回带有结构化信息的 list；默认值为 `True`；
* 输入参数 `rec_config`：识别配置，可选值；
* 输入参数 `kwargs`：其他参数，具体说明参考函数 `text_ocr.ocr()`。

**返回值**：当 `return_text` 为 `True` 时，返回一个字符串；当 `return_text` 为 `False` 时，返回一个有序的（从上到下，从左到右）字典列表，每个字典表示一个检测框，包含以下键值：
	- `position`：块的位置信息，`np.ndarray`，形状为 `[4, 2]`
	- `text`：识别的文本
	- `score`：置信度分数 [0, 1]；分数越高，置信度越高

**调用示例**：

```python
from pix2text import Pix2Text

img_fp = 'examples/general.jpg'
p2t = Pix2Text.from_config()
out = p2t.recognize_text(img_fp)
```

### 6. 函数 `.recognize()`

是不是觉得上面的接口太丰富了，使用起来有点麻烦？没关系，这个函数可以根据指定的图片类型调用上面的不同函数进行识别。

```python
def recognize(
		self,
		img: Union[str, Path, Image.Image],
		file_type: Literal[
			'pdf', 'page', 'text_formula', 'formula', 'text'
		] = 'text_formula',
		**kwargs,
) -> Union[Document, Page, str, List[str], List[Any], List[List[Any]]]:
	"""
    Recognize the content of the image or pdf file according to the specified type.
    It will call the corresponding recognition function `.recognize_{file_type}()` according to the `file_type`.
    Args:
        img (Union[str, Path, Image.Image]): The image/pdf file path or `Image.Image` object
        file_type (str):  Supported image types: 'pdf', 'page', 'text_formula', 'formula', 'text'
        **kwargs (dict): Arguments for the corresponding recognition function

    Returns: recognized results

    """
```

**函数说明**：

* 输入参数 `img`：图片/PDF文件路径或者 `Image.Image` 对象；
* 输入参数 `file_type`：图片类型，可选值为 `'pdf'`, `'page'`, `'text_formula'`, `'formula'`, `'text'`；
* 输入参数 `kwargs`：其他参数，具体说明参考上面的函数。

**返回值**：根据 `file_type` 的不同，返回不同的结果。具体说明参考上面的函数。

**调用示例**：

```python
from pix2text import Pix2Text

img_fp = 'examples/general.jpg'
p2t = Pix2Text.from_config()
out = p2t.recognize(img_fp, file_type='text')  # 等价于 p2t.recognize_text(img_fp)
```


更多使用示例请参见 [tests/test_pix2text.py](https://github.com/breezedeus/Pix2Text/blob/main/tests/test_pix2text.py)。
