<div align="center">
  <img src="./docs/figs/p2t.jpg" width="250px"/>
  <div>&nbsp;</div>

[![Downloads](https://static.pepy.tech/personalized-badge/pix2text?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/pix2text)
[![license](https://img.shields.io/github/license/breezedeus/pix2text)](./LICENSE)
[![PyPI version](https://badge.fury.io/py/pix2text.svg)](https://badge.fury.io/py/pix2text)
[![forks](https://img.shields.io/github/forks/breezedeus/pix2text)](https://github.com/breezedeus/pix2text)
[![stars](https://img.shields.io/github/stars/breezedeus/pix2text)](https://github.com/breezedeus/pix2text)
![last-release](https://img.shields.io/github/release-date/breezedeus/pix2text)
![last-commit](https://img.shields.io/github/last-commit/breezedeus/pix2text)
[![Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fbreezedeus)](https://twitter.com/breezedeus)

[👩🏻‍💻 Online Demo](https://p2t.breezedeus.com) |
[💬 Contact](https://www.breezedeus.com/join-group)

</div>

<div align="center">

[中文](./README_cn.md) | English

</div>



# Pix2Text

## Update 2023.07.03: Released V0.2.3

Major changes:

- Trained a new **formula recognition model** for **[P2T Online Service](https://p2t.breezedeus.com/)** to use. The new model has higher accuracy, especially for **handwritten formulas** and **multi-line formulas**. See: [New Formula Recognition Model for Pix2Text | Breezedeus.com](https://www.breezedeus.com/article/p2t-mfd-20230702).
- Optimized the sorting logic of detected boxes and the processing logic of mixed images to make the final recognition results more intuitive.
- Optimized the merging logic of recognition results to automatically determine line breaks and paragraph breaks.
- Fixed the automatic model downloading feature. HuggingFace seems to have changed the downloading logic, which caused the previous version's auto-download to fail. The current version has fixed it. 
- Updated the version numbers of various dependency packages.

## Update 2023.06.20: Released New MFD Model

Major changes:

- Retrained the **MFD YoloV7** model on new annotated data. The new model has been deployed to [P2T Online Service](https://p2t.breezedeus.com/). See: [New Formula Detection Model for Pix2Text (P2T) | Breezedeus.com](https://www.breezedeus.com/article/p2t-mfd-20230613).
- The previous MFD YoloV7 model is now available for download to community members. See: [P2T YoloV7 Formula Detection Model Released to Community Members | Breezedeus.com](https://www.breezedeus.com/article/p2t-yolov7-for-zsxq-20230619).

## Update 2023.02.10: **[P2T Online Service](https://p2t.breezedeus.com/)** Open for Free Use

- As a Python package, Pix2Text is not very beginner-friendly. So we also developed the [P2T Online Service](https://p2t.breezedeus.com/) that can be used for free directly. Feel free to help spread the word!
- Video intro: [Pix2Text New Version and Web Version Released, Getting Closer to Mathpix_bilibili](https://www.bilibili.com/video/BV1U24y1q7n3)
- Text intro: [Pix2Text New Version Released, Getting Closer to Mathpix - Zhihu](https://zhuanlan.zhihu.com/p/604999678)

See more at: [RELEASE.md](./RELEASE.md) .



**Pix2Text** aims to be a **free and open-source Python** alternative to **[Mathpix](https://mathpix.com/)**. It can already complete the core functionalities of **Mathpix**. Starting from **V0.2**, **Pix2Text (P2T)** supports recognizing **mixed images containing both text and formulas**, with output similar to **Mathpix**. The core principles of P2T are shown below (text recognition supports both **Chinese** and **English**):

<div align="center"> <img src="./docs/figs/arch-flow2.jpg" alt="Pix2Text workflow" width="600px"/> </div>

**P2T** uses the open-source tool [**CnSTD**](https://github.com/breezedeus/cnstd) to detect **formula** regions in the image. The formulas are then fed into [**LaTeX-OCR**](https://github.com/lukas-blecher/LaTeX-OCR) to recognize their LaTeX expressions. The remaining text regions are recognized by [**CnOCR**](https://github.com/breezedeus/cnocr). Finally, P2T merges all results to get the full recognized texts. Thanks to these great open-source projects!

For beginners who are not familiar with Python, we also provide the **free-to-use** [P2T Online Service](https://p2t.breezedeus.com/). Just upload your image and it will output the P2T parsing results. **The online service uses the latest models and works better than the open-source ones.**

If interested, please scan the QR code below to add the assistant WeChat account, and send `p2t` to get invited to the P2T user group. The group shares the latest updates of P2T and related tools:

<div align="center"> <img src="./docs/figs/wx-qr-code.JPG" alt="WeChat Group QR Code" width="300px"/> </div>



The author also maintains **Planet of Knowledge** [**P2T/CnOCR/CnSTD Private Group**](https://t.zsxq.com/FEYZRJQ), welcome to join. The **Planet of Knowledge Private Group** will release some P2T/CnOCR/CnSTD related private materials one after another, including **non-public models**, **discount for paid models**, answers to problems encountered during usage, etc. This group also releases the latest research materials related to VIE/OCR/STD.



## Usage


Pix2Text is very simple to use and the following is an example:

```python
from pix2text import Pix2Text, merge_line_texts

img_fp = './docs/examples/formula.jpg'
p2t = Pix2Text(analyzer_config=dict(model_name='mfd'))
outs = p2t(img_fp, resized_shape=600)  # # can also use `p2t.recognize(img_fp)`
print(outs)
# To get just the text contents, use: 
only_text = merge_line_texts(outs, auto_line_break=True)
print(only_text)
```

The returned `outs` is a `dict` where `position` gives the box coordinates, `type` the predicted type, and `text` the recognized texts. See [API Interfaces](#接口说明) for details.


Some examples:

<table>
<tr>
<th> Image </th> 
<th> Pix2Text's Result </th>
</tr>
<tr>
<td>
<img src="./docs/examples/mixed.jpg" alt="mixed"> 

</td>
<td>

```python
[{'line_number': 0,
  'position': array([[         22,          31],
       [       1057,          31],
       [       1057,          58],
       [         22,          58]]),
  'text': 'JVAE的训练loss和VQ-VAE类似，只是使用了KL距离来让分布尽量分散',
  'type': 'text'},
 {'line_number': 1,
  'position': array([[        625,         121],
       [       1388,         121],
       [       1388,         182],
       [        625,         182]]),
  'text': '$$\n'
          '-E_{z\\sim q(z\\mid x)}[\\log(p(x\\mid z))]+K L(q(z\\mid x))|p(z))\n'
          '$$',
  'type': 'isolated'},
 {'line_number': 2,
  'position': array([[         18,         242],
       [        470,         242],
       [        470,         275],
       [         18,         275]]),
  'text': '其中之利用 Gumbel-Softmax 人',
  'type': 'text'},
 {'line_number': 2,
  'position': array([[        481,         238],
       [        664,         238],
       [        664,         287],
       [        481,         287]]),
  'text': ' $z\\sim q(z|x)$ ',
  'type': 'embedding'},
 {'line_number': 2,
  'position': array([[        667,         244],
       [        840,         244],
       [        840,         277],
       [        667,         277]]),
  'text': '中抽样得到,',
  'type': 'text'},
 {'line_number': 2,
  'position': array([[        852,         239],
       [        932,         239],
       [        932,         288],
       [        852,         288]]),
  'text': ' $\\scriptstyle{p(z)}$ ',
  'type': 'embedding'},
 {'line_number': 2,
  'position': array([[        937,         244],
       [       1299,         244],
       [       1299,         277],
       [        937,         277]]),
  'text': '是个等概率的多项式分布',
  'type': 'text'}]
```

</td>
</tr>
<tr>
<td>

<img src="./docs/examples/formula.jpg" alt="formula"> 
</td>
<td>

```python
[{"line_number": 0,
  "position": array([[         12,          19],
       [        749,          19],
       [        749,         150],
       [         12,         150]]),
  "text": "$$\n"
          "\\mathcal{L}_{\\mathrm{eyelid}}~\\equiv~"
          "\\sum_{t=1}^{T}\\sum_{v=1}^{V}"
          "\\mathcal{N}_{U}^{\\mathrm{(eyelid)}}"
          "\\left(\\left|\\left|\\hat{h}_{t,v}\\,-\\,"
          "\\mathcal{x}_{t,v}\\right|\\right|^{2}\\right)\n"
          "$$",
  "type": "isolated"}]
```
</div>
</td>
</tr>
<tr>
<td>

 <img src="./docs/examples/english.jpg" alt="english"> 
</td>
<td>

```python
[{"position": array([[          0,           0],
       [        710,           0],
       [        710,         116],
       [          0,         116]]),
  "text": "python scripts/screenshot_daemon_with_server\n"
          "2-get_model:178usemodel:/Users/king/.cr\n"
          "enet_lite_136-fc-epoch=039-complete_match_er",
  "type": "english"}]
```
</td>
</tr>
<tr>
<td>

 <img src="./docs/examples/general.jpg" alt="general"  width="300px"> 
</td>
<td>

```python
[{"position": array([[          0,           0],
       [        800,           0],
       [        800,         800],
       [          0,         800]]),
  "text": "618\n开门红提前购\n很贵\n买贵返差"
  "\n终于降价了\n100%桑蚕丝\n要买趁早\n今日下单188元\n仅限一天",
  "type": "general"}]
```
</td>
</tr>
</table>



### Model Download

#### Free Open-source Models

After installing Pix2Text, the system will **automatically download** the model files and store them in `~/.pix2text` directory when you use Pix2Text for the first time (the default path under Windows is `C:\Users\<username>\AppData\Roaming\pix2text`).

> **Note**
>
> If you have successfully run the above example, the model has completed its automatic download and you can ignore the subsequent contents of this section.

For the **classifier model**, the system will automatically download the model file `mobilenet_v2.zip` and unzip it, putting the extracted model directories under `~/.pix2text`. If it fails, you need to manually download the `mobilenet_v2.zip` file from [**cnstd-cnocr-models/pix2text**](https://huggingface.co/breezedeus/cnstd-cnocr-models/tree/main/models/pix2text/0.2) and put it under `~/.pix2text`. If the download is too slow, you can also download it from [Baidu Cloud](https://pan.baidu.com/s/1kubZF4JGE19d98NDoPHJzQ?pwd=p2t0) with code `p2t0`.

For [**LaTeX-OCR**](https://github.com/lukas-blecher/LaTeX-OCR), the system will also try to automatically download its model files `weights.pth` and `image_resizer.pth` into `~/.pix2text/formula`. If failed, you need to download them from [Baidu Cloud](https://pan.baidu.com/s/1kubZF4JGE19d98NDoPHJzQ?pwd=p2t0) and put them under `~/.pix2text/formula`; code: `p2t0`.



#### Paid Models

In addition to the above free open-source models, we also trained higher-accuracy formula detection and recognition models for P2T. They are used by the **[P2T Online Service](https://p2t.breezedeus.com/)** on which you can try the performance. These models are not free (sorry open-source developers need coffee too🥤). See [Pix2Text (P2T) | Breezedeus.com](https://www.breezedeus.com/pix2text) for details.




## Install

Well, one line of command is enough if it goes well.

```bash
pip install pix2text
```

If the installation is slow, you can specify a domestic installation source, such as using the Douban source: 

```bash
pip install pix2text -i https://pypi.doubanio.com/simple
```


If it is your first time to use **OpenCV**, then probably  the installation will not be very easy.  Bless.

**Pix2Text** mainly depends on [**CnOCR>=2.2.2**](https://github.com/breezedeus/cnocr) , and [**LaTeX-OCR**](https://github.com/lukas-blecher/LaTeX-OCR). If you encounter problems with the installation, you can also refer to their installation instruction documentations.


> **Warning** 
>
> If you have never installed the `PyTorch`, `OpenCV` python packages before, you may encounter a lot of problems during the first installation, but they are usually common problems that can be solved by Baidu/Google.

## API Interfaces

### Class Initializer

Main class called [**Pix2Text**](pix2text/pix_to_text.py) , with initialization function:

```python
class Pix2Text(object):

    def __init__(
        self,
        *,
        analyzer_config: Dict[str, Any] = None,
        clf_config: Dict[str, Any] = None,
        general_config: Dict[str, Any] = None,
        english_config: Dict[str, Any] = None,
        formula_config: Dict[str, Any] = None,
        thresholds: Dict[str, Any] = None,
        device: str = 'cpu',  # ['cpu', 'cuda', 'gpu']
        **kwargs,
    ):
```



The parameters are described as follows:

- `analyzer_config` (dict): Configuration for the classifier model. Default to `None` meaning using default config (MFD Analyzer):

  ```python
  {
      'model_name': 'mfd' # can be 'mfd' or 'layout'
  }
  ```

- `clf_config` (dict): Configuration for the classifier model. Default to `None` meaning using default:

  ```python
  {
      'base_model_name': 'mobilenet_v2',
      'categories': IMAGE_TYPES,
      'transform_configs': {
          'crop_size': [150, 450],
          'resize_size': 160,
          'resize_max_size': 1000,
      },
      'model_dir': Path(data_dir()) / 'clf',
      'model_fp': None # use this model file if specified
  }
  ```

- `general_config` (dict): Configuration for the general recognizer. Default to `None` meaning using default:

  ```python
  {}
  ```

- `english_config` (dict): Configuration for the English recognizer. Default to `None` meaning using default:

  ```python
  {'det_model_name': 'en_PP-OCRv3_det', 'rec_model_name': 'en_PP-OCRv3'}
  ```

- `formula_config` (dict): Configuration for the formula recognizer. Default to `None` meaning using default:

  ```python
  {
      'config': LATEX_CONFIG_FP,
      'model_fp': Path(data_dir()) / 'formula' / 'weights.pth',
      'no_resize': False
  }
  ```

- `thresholds` (dict): Thresholds for prediction confidence. Default to `None` meaning using default:

  ```python
  {
      'formula2general': 0.65, # Lower confidence formula -> general
      'english2general': 0.75, # Lower confidence english -> general 
  }
  ```

- `device` (str): Device for running the code, can be `['cpu', 'cuda', 'gpu']`. Default: `'cpu'`

- `**kwargs` (): Other reserved parameters. Currently not used.



### Class Function for Recognition

The text or Latex recognition of one specified image is done by invoking the class function `.recognize()` of class **`Pix2Text`**. The class function `.recognize()` is described as follows.

```py
    def recognize(
        self, img: Union[str, Path, Image.Image], use_analyzer: bool = True, **kwargs
    ) -> List[Dict[str, Any]]:
```

where the input parameters are described as follows.

* `img` (`str` or `Image.Image`): the path of the image to be recognized, or the image `Image` that has been read by using `Image.open()`.

* `use_analyzer`: Whether to use the Analyzer (MFD or Layout). `False` means treat the image as pure text or math.

* `**kwargs`: Can contain:
  - `resized_shape`: Resize image width to this before processing. Default: `700`.
  - `save_analysis_res`: Save analysis visualization to this file/dir. Default: `None` meaning not saving.
  - `embed_sep`: LaTeX delimiter for embedded formulas. Only useful with MFD. Default: `(' $', '$ ')`.
  - `isolated_sep`: LaTeX delimiter for isolated formulas. Only useful with MFD. Default: `('$$\n', '\n$$')`.

It returns a `list` of `dict`, each `dict` contains:

- `type`: Predicted type, can be:

  - `text`, `isolated`, `embedding` when `use_analyzer==True`.

    > Note: The values are different from P2T **v0.2.3** and before when using **MFD Analyzer**.

  - `formula`, `english`, `general` when `use_analyzer==False`.

- `text`: Recognized text or latex.

- `position`: Detected box coordinates, `np.ndarray`, with shape `[4, 2]`.

- `line_number`: Exists only when using **MFD Analyzer**. Indicates the line number (starting from 0) of the box. Boxes with the same `line_number` are on the same line.

  > Note: This is new since P2T **v0.2.3**. Not in previous versions.



The `Pix2Text` class also implements the `__call__()` function, which does exactly the same thing as the `.recognize()` function.  So you can call it like:

```python
from pix2text import Pix2Text, merge_line_texts

img_fp = './docs/examples/formula.jpg'
p2t = Pix2Text(analyzer_config=dict(model_name='mfd'))
outs = p2t(img_fp, resized_shape=608) # Equal to p2t.recognize()
print(outs)
# To get just the text contents, use: 
only_text = merge_line_texts(outs, auto_line_break=True)
print(only_text)
```



## Script Usage

**P2T** includes the following command-line tools.

### Recognizing a single image or all images in a directory

Use the **`p2t predict`** command to predict a single image or all images in a directory. Below is the usage guide:

```bash
$ p2t predict -h
Usage: p2t predict [OPTIONS]

  Model prediction

Options:
  --use-analyzer / --no-use-analyzer
                                  Whether to use MFD or layout analysis Analyzer  [default: use-
                                  analyzer]
  -a, --analyzer-name [mfd|layout]
                                  Which Analyzer to use, MFD or layout analysis  [default: mfd]
  -t, --analyzer-type TEXT        Which model should the Analyzer use, 'yolov7_tiny' or 'yolov7'
                                  [default: yolov7_tiny]
  --analyzer-model-fp TEXT        File path for the Analyzer detection model. Default: `None`, meaning to use the default model
  --latex-ocr-model-fp TEXT       File path for the Latex-OCR
                                  mathematical formula recognition model. Default: `None`, indicating using the default model
  -d, --device TEXT               Use `cpu` or `gpu` to run the code, or specify a particular GPU, such as `cuda:0`
                                  [default: cpu]
  --resized-shape INTEGER         Resize the image width to this size for processing  [default: 608]
  -i, --img-file-or-dir TEXT      Input path for the image file or specified directory [required]
  --save-analysis-res TEXT        Save the analysis results to this file or directory (if '--img-file-or-
                                  dir' is a file/directory, then '--save-analysis-
                                  res' should also be a file/directory). A value of `None` means not to save
  -l, --log-level TEXT            Log Level, such as `INFO`, `DEBUG`
                                  [default: INFO]
  -h, --help                      Show this message and exit.
```

This command can be used to **print detection and recognition results for the specified image**. For example, run:

```bash
$ p2t predict --use-analyzer -a mfd --resized-shape 608 -i docs/examples/en1.jpg --save-analysis-res output-en1.jpg
```

The above command prints the recognition results, and it will also store the detection results in the `output-en1.jpg` file, similar to the effect below:

<div align="center">
  <img src="./docs/figs/output-en1.jpg" alt="P2T Mathematical Formula Detection Effect Image" width="600px"/>
</div>



## HTTP Server

 **Pix2Text** adds the FastAPI-based HTTP server. The server requires the installation of several additional packages, which can be installed using the following command.

```bash
> pip install pix2text[serve]
```

Once the installation is complete, the HTTP server can be started with the following command (**`-p`** followed by the **port**, which can be adjusted as needed).


```bash
> p2t serve -p 8503
```



`p2t serve` command usage guide:

```bash
$ p2t serve -h
Usage: p2t serve [OPTIONS]

  Start the HTTP service.

Options:
  -H, --host TEXT     server host  [default: 0.0.0.0]
  -p, --port INTEGER  server port  [default: 8503]
  --reload            whether to reload the server when the codes have been
                      changed
  -h, --help          Show this message and exit.
```

After the service starts, you can call the service in the following ways.




### Command Line

As an example, if the file to be recognized is `docs/examples/mixed.jpg`, use `curl` to invoke the server:

```bash
$ curl -F image=@docs/examples/mixed.jpg --form 'use_analyzer=true' --form 'resized_shape=600' http://0.0.0.0:8503/pix2text
```


### Python

To call the service, refer to the following method in the file [scripts/try_service.py](scripts/try_service.py):

```python
import requests

url = 'http://0.0.0.0:8503/pix2text'

image_fp = 'docs/examples/mixed.jpg'
data = {
    "use_analyzer": True,
    "resized_shape": 608,
    "embed_sep": " $,$ ",
    "isolated_sep": "$$\n, \n$$"
}
files = {
    "image": (image_fp, open(image_fp, 'rb'))
}

r = requests.post(url, data=data, files=files)

outs = r.json()['results']
only_text = '\n'.join([out['text'] for out in outs])
print(f'{only_text=}')
```





### Other Language

Please refer to the `curl` format for your own implementation.


## Use Script

Script [scripts/screenshot_daemon.py](scripts/screenshot_daemon.py) automatically invokes Pixe2Text to recognize formulas or texts on screenshot images. How does this work?



**Here's the process (please install Pix2Text first):**

1. Find one favorite screenshot tool that **supports storing screenshot images in a specified folder**. For example, the free **Xnip** for Mac works very well.

2. In addition to installing Pix2Text, you need to install an additional Python package **pyperclip**, which you can use to copy the recognition results into the system clipboard: 

   ```bash
   $ pip install pyperclip
   ```

3. Download the script file [scripts/screenshot_daemon.py](scripts/screenshot_daemon.py) to your computer, edit the line where `"SCREENSHOT_DIR"` is located (line `17`) and change the path to the directory where your screenshot images are stored.

4. Run this script.

   ```bash
   $ python scripts/screenshot_daemon.py
   ```


Alright, now give it a shot using your screenshot software. Once you've taken the screenshot, the recognition results will be written to your computer's clipboard. Simply press **Ctrl-V** / **Cmd-V** to paste and use it.



For a more detailed introduction, please refer to the video: "[Pix2Text: A Free Python Open Source Tool to Replace Mathpix](https://www.bilibili.com/video/BV12e4y1871U)".




## A cup of coffee for the author

It is not easy to maintain and evolve the project, so if it is helpful to you, please consider [offering the author a cup of coffee 🥤](https://www.breezedeus.com/buy-me-coffee).

---

Official code base: [https://github.com/breezedeus/pix2text](https://github.com/breezedeus/pix2text). Please cite it properly.

For more information on Pix2Text (P2T), visit: [https://www.breezedeus.com/pix2text](https://www.breezedeus.com/pix2text).
