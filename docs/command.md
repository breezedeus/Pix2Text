# 脚本工具

Python 包 **pix2text** 自带了命令行工具 `p2t`，[安装](install.md) 后即可使用。`p2t` 包含了以下几个子命令。

## 预测

使用命令 **`p2t predict`** 预测单个（图片或 PDF）文件或文件夹中所有图片（不支持同时预测多个 PDF 文件），以下是使用说明：

```bash
$ p2t predict -h
Usage: p2t predict [OPTIONS]

  使用Pix2Text（P2T）来预测图像或 PDF 文件中的文本信息

选项：
  -l，--languages TEXT            Text-OCR识别的语言代码，用逗号分隔，默认为en,ch_sim
  --layout-config TEXT            布局解析器模型的配置信息，以JSON字符串格式提供。默认值：`None`，表示使用默认配置
  --mfd-config TEXT               MFD模型的配置信息，以JSON字符串格式提供。默认值：`None`，表示使用默认配置
  --formula-ocr-config TEXT       Latex-OCR数学公式识别模型的配置信息，以JSON字符串格式提供。默认值：`None`，表示使用默认配置
  --text-ocr-config TEXT          Text-OCR识别的配置信息，以JSON字符串格式提供。默认值：`None`，表示使用默认配置
  --enable-formula / --disable-formula
                                  是否启用公式识别，默认值：启用公式
  --enable-table / --disable-table
                                  是否启用表格识别，默认值：启用表格
  -d, --device TEXT               选择使用`cpu`、`gpu`或指定的GPU，如`cuda:0`。默认值：cpu
  --file-type [pdf|page|text_formula|formula|text]
                                  要处理的文件类型，'pdf'、'page'、'text_formula'、'formula'或'text'。默认值：text_formula
  --resized-shape INTEGER         在处理之前将图像宽度调整为此大小。默认值：768
  -i, --img-file-or-dir TEXT      输入图像/pdf的文件路径或指定的目录。[必需]
  --save-debug-res TEXT           如果设置了`save_debug_res`，则保存调试结果的目录；默认值为`None`，表示不保存
  --rec-kwargs TEXT               用于调用`.recognize()`的kwargs，以JSON字符串格式提供
  --return-text / --no-return-text
                                  是否仅返回文本结果，默认值：返回文本
  --auto-line-break / --no-auto-line-break
                                  是否自动确定是否将相邻的行结果合并为单个行结果，默认值：自动换行
  -o, --output-dir TEXT           识别文本结果的输出目录。仅在`file-type`为`pdf`或`page`时有效。默认值：output-md
  --log-level TEXT                日志级别，例如`INFO`、`DEBUG`。默认值：INFO
  -h, --help                      显示此消息并退出。
```

### 示例 1
使用基础模型进行预测：

```bash
p2t predict -l en,ch_sim --resized-shape 768 --file-type pdf -i docs/examples/test-doc.pdf -o output-md --save-debug-res output-debug
```

它会把识别结果（Markdown格式）存放在 `output-md` 目录下，并把中间的解析结果存放在 `output-debug` 目录下，以便分析识别结果主要受哪个模型的影响。
如果不需要保存中间解析结果，可以去掉 `--save-debug-res output-debug` 参数。

### 示例 2

预测时也支持使用自定义的参数或模型。例如，使用自定义的模型进行预测：

```bash
p2t predict -l en,ch_sim --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --rec-kwargs '{"page_numbers": [0, 1]}' --resized-shape 768 --file-type pdf -i docs/examples/test-doc.pdf -o output-md --save-debug-res output-debug
```


## 开启服务

使用命令 **`p2t serve`** 开启一个 HTTP 服务，用于接收图片（当前不支持 PDF）并返回识别结果。
这个 HTTP 服务是基于 FastAPI 实现的，以下是使用说明：

```bash
$ p2t serve -h
Usage: p2t serve [OPTIONS]

  启动HTTP服务。

选项：
  -l, --languages TEXT            Text-OCR识别的语言代码，用逗号分隔，默认为en,ch_sim
  --layout-config TEXT            布局解析器模型的配置信息，以JSON字符串格式提供。默认值：`None`，表示使用默认配置
  --mfd-config TEXT               MFD模型的配置信息，以JSON字符串格式提供。默认值：`None`，表示使用默认配置
  --formula-ocr-config TEXT       Latex-OCR数学公式识别模型的配置信息，以JSON字符串格式提供。默认值：`None`，表示使用默认配置
  --text-ocr-config TEXT          Text-OCR识别的配置信息，以JSON字符串格式提供。默认值：`None`，表示使用默认配置
  --enable-formula / --disable-formula
                                  是否启用公式识别，默认值：启用公式
  --enable-table / --disable-table
                                  是否启用表格识别，默认值：启用表格
  -d, --device TEXT               选择使用`cpu`、`gpu`或指定的GPU，如`cuda:0`。默认值：cpu
  -o, --output-md-root-dir TEXT   Markdown输出的根目录，用于存放识别文本结果。仅在`file-type`为`pdf`或`page`时有效。默认值：output-md-root
  -H, --host TEXT                 服务器主机  [默认值：0.0.0.0]
  -p, --port INTEGER              服务器端口  [默认值：8503]
  --reload                        当代码发生更改时是否重新加载服务器
  --log-level TEXT                日志级别，例如`INFO`、`DEBUG`。默认值：INFO
  -h, --help                      显示此消息并退出。
```

### 示例 1
使用基础模型进行预测：

```bash
p2t serve -l en,ch_sim -H 0.0.0.0 -p 8503
```

### 示例 2

服务开启时也支持使用自定义的参数或模型。例如，使用自定义的模型进行预测：

```bash
p2t serve -l en,ch_sim --mfd-config '{"model_type": "yolov7", "model_fp": "/Users/king/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' -H 0.0.0.0 -p 8503
```

### 服务调用

#### Python
开启后可以使用以下方式调用命令（Python）：

```python
import requests

url = 'http://0.0.0.0:8503/pix2text'

image_fp = 'docs/examples/page2.png'
data = {
    "file_type": "page",
    "resized_shape": 768,
    "embed_sep": " $,$ ",
    "isolated_sep": "$$\n, \n$$"
}
files = {
    "image": (image_fp, open(image_fp, 'rb'), 'image/jpeg')
}

r = requests.post(url, data=data, files=files)

outs = r.json()['results']
out_md_dir = r.json()['output_dir']
if isinstance(outs, str):
    only_text = outs
else:
    only_text = '\n'.join([out['text'] for out in outs])
print(f'{only_text=}')
print(f'{out_md_dir=}')
```

#### Curl

也可以使用 curl 调用服务：

```bash
curl -X POST \
  -F "file_type=page" \
  -F "resized_shape=768" \
  -F "embed_sep= $,$ " \
  -F "isolated_sep=$$\n, \n$$" \
  -F "image=@docs/examples/page2.png;type=image/jpeg" \
  http://0.0.0.0:8503/pix2text
```