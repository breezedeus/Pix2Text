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

[ðð» å¨çº¿Demo](https://huggingface.co/spaces/breezedeus/pix2text) |
[ð¬ äº¤æµç¾¤](https://cnocr.readthedocs.io/zh/latest/contact/)

</div>

<div align="center">

[English](./README_en.md) | ä¸­æ
</div>

# Pix2Text



**Pix2Text** æææä¸º **[Mathpix](https://mathpix.com/)** ç**åè´¹å¼æº Python **æ¿ä»£å·¥å·ï¼å®æä¸ Mathpix ç±»ä¼¼çåè½ãå½å Pix2Text å¯è¯å«æªå±å¾çä¸­ç**æ°å­¦å¬å¼**ã**è±æ**ãæè**ä¸­ææå­**ãå®çæµç¨å¦ä¸ï¼

<div align="center">
  <img src="./docs/figs/arch-flow.jpg" alt="Pix2Textæµç¨" width="800px"/>
</div>



Pix2Texté¦åå©ç¨**å¾çåç±»æ¨¡å**æ¥å¤æ­å¾çç±»åï¼ç¶ååºäºä¸åçå¾çç±»åï¼æå¾çäº¤ç±ä¸åçè¯å«ç³»ç»è¿è¡æå­è¯å«ï¼

1. å¦æå¾çç±»åä¸º `formula` ï¼è¡¨ç¤ºå¾çä¸ºæ°å­¦å¬å¼ï¼æ­¤æ¶è°ç¨ [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR) è¯å«å¾çä¸­çæ°å­¦å¬å¼ï¼è¿åå¶Latexè¡¨ç¤ºï¼
1. å¦æå¾çç±»åä¸º`english`ï¼è¡¨ç¤ºå¾çä¸­åå«çæ¯è±ææå­ï¼æ­¤æ¶ä½¿ç¨ [CnOCR](https://github.com/breezedeus/cnocr) ä¸­ç**è±ææ¨¡å**è¯å«å¶ä¸­çè±ææå­ï¼è±ææ¨¡åå¯¹äºçº¯è±æçæå­æªå¾ï¼è¯å«æææ¯éç¨æ¨¡åå¥½ï¼
1. å¦æå¾çç±»åä¸º`general`ï¼è¡¨ç¤ºå¾çä¸­åå«çæ¯å¸¸è§æå­ï¼æ­¤æ¶ä½¿ç¨ [CnOCR](https://github.com/breezedeus/cnocr) ä¸­ç**éç¨æ¨¡å**è¯å«å¶ä¸­çä¸­æè±ææå­ã



åç»­å¾çç±»åä¼ä¾æ®åºç¨éè¦åè¿ä¸æ­¥çç»åã



æ¬¢è¿æ«ç å å°å©æä¸ºå¥½åï¼å¤æ³¨ `p2t`ï¼å°å©æä¼å®æç»ä¸éè¯·å¤§å®¶å¥ç¾¤ï¼

<div align="center">
  <img src="./docs/figs/wx-qr-code.JPG" alt="å¾®ä¿¡ç¾¤äºç»´ç " width="300px"/>
</div>



ä½èä¹ç»´æ¤ **ç¥è¯æç** [**P2T/CnOCR/CnSTDç§äº«ç¾¤**](https://t.zsxq.com/FEYZRJQ) ï¼è¿éé¢çæé®ä¼è¾å¿«å¾å°ä½èçåå¤ï¼æ¬¢è¿å å¥ã**ç¥è¯æçç§äº«ç¾¤**ä¹ä¼éç»­åå¸ä¸äºP2T/CnOCR/CnSTDç¸å³çç§æèµæï¼åæ¬[**æ´è¯¦ç»çè®­ç»æç¨**](https://articles.zsxq.com/id_u6b4u0wrf46e.html)ï¼**æªå¬å¼çæ¨¡å**ï¼**ä¸ååºç¨åºæ¯çè°ç¨ä»£ç **ï¼ä½¿ç¨è¿ç¨ä¸­éå°çé¾é¢è§£ç­ç­ãæ¬ç¾¤ä¹ä¼åå¸OCR/STDç¸å³çææ°ç ç©¶èµæã



## ä½¿ç¨è¯´æ


è°ç¨å¾ç®åï¼ä»¥ä¸æ¯ç¤ºä¾ï¼

```python
from pix2text import Pix2Text

img_fp = './docs/examples/formula.jpg'
p2t = Pix2Text()
out_text = p2t(img_fp)  # ä¹å¯ä»¥ä½¿ç¨ `p2t.recognize(img_fp)` è·å¾ç¸åçç»æ
print(out_text)
```



è¿åç»æ `out_text` æ¯ä¸ª `dict`ï¼å¶ä¸­ key `image_type` è¡¨ç¤ºå¾çåç±»ç±»å«ï¼è key `text` è¡¨ç¤ºè¯å«çç»æã



ä»¥ä¸æ¯ä¸äºç¤ºä¾å¾ççè¯å«ç»æï¼

<table>
<tr>
<td> å¾ç </td> <td> Pix2Textè¯å«ç»æ </td>
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
 "text": "618\nå¼é¨çº¢æåè´­\nå¾è´µ\nä¹°è´µè¿å·®\nç»äºéä»·äº\n"
          "100%æ¡èä¸\nè¦ä¹°è¶æ©\nä»æ¥ä¸å188å\nä»éä¸å¤©"}
```
</td>
</tr>
</table>



### æ¨¡åä¸è½½

å®è£å¥½ Pix2Text åï¼é¦æ¬¡ä½¿ç¨æ¶ç³»ç»ä¼**èªå¨ä¸è½½** æ¨¡åæä»¶ï¼å¹¶å­äº `~/.pix2text`ç®å½ï¼Windowsä¸é»è®¤è·¯å¾ä¸º `C:\Users\<username>\AppData\Roaming\pix2text`ï¼ã



> **Note**
>
> å¦æå·²æåè¿è¡ä¸é¢çç¤ºä¾ï¼è¯´ææ¨¡åå·²å®æèªå¨ä¸è½½ï¼å¯å¿½ç¥æ¬èåç»­åå®¹ã



å¯¹äº**åç±»æ¨¡å**ï¼ç³»ç»ä¼èªå¨ä¸è½½æ¨¡åzipæä»¶å¹¶å¯¹å¶è§£åï¼ç¶åæè§£ååçæ¨¡åç¸å³ç®å½æ¾äº`~/.pix2text`ç®å½ä¸­ãå¦æç³»ç»æ æ³èªå¨æåä¸è½½zipæä»¶ï¼åéè¦æå¨ä» **[cnstd-cnocr-models/pix2text](https://huggingface.co/breezedeus/cnstd-cnocr-models/tree/main/models/pix2text/0.1)** ä¸è½½æ­¤zipæä»¶å¹¶æå®æ¾äº `~/.pix2text`ç®å½ãå¦æä¸è½½å¤ªæ¢ï¼ä¹å¯ä»¥ä» [ç¾åº¦äºç](https://pan.baidu.com/s/10E_NAAWHnbcCu7tw3vnbjg?pwd=p2t0) ä¸è½½ï¼ æåç ä¸º ` p2t0`ã

å¯¹äº  **[LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)** ï¼ç³»ç»åæ ·ä¼èªå¨ä¸è½½æ¨¡åæä»¶å¹¶æå®ä»¬å­æ¾äº`~/.pix2text/formula`ç®å½ä¸­ãå¦æç³»ç»æ æ³èªå¨æåä¸è½½è¿äºæ¨¡åæä»¶ï¼åéä»  [ç¾åº¦äºç](https://pan.baidu.com/s/1KgFLm6iTRK0Zn8fvu2aDzQ?pwd=p2t0) ä¸è½½æä»¶ `weights.pth` å `image_resizer.pth`ï¼ å¹¶æå®ä»¬å­æ¾äº`~/.pix2text/formula`ç®å½ä¸­ï¼æåç ä¸º ` p2t0`ã



## å®è£

å¯ï¼é¡ºå©çè¯ä¸è¡å½ä»¤å³å¯ã

```bash
pip install pix2text
```

å®è£éåº¦æ¢çè¯ï¼å¯ä»¥æå®å½åçå®è£æºï¼å¦ä½¿ç¨è±ç£æºï¼

```bash
pip install pix2text -i https://pypi.doubanio.com/simple
```



å¦ææ¯åæ¬¡ä½¿ç¨**OpenCV**ï¼é£ä¼°è®¡å®è£é½ä¸ä¼å¾é¡ºå©ï¼blessã

**Pix2Text** ä¸»è¦ä¾èµ [**CnOCR>=2.2.2**](https://github.com/breezedeus/cnocr) ï¼ä»¥å [**LaTeX-OCR**](https://github.com/lukas-blecher/LaTeX-OCR) ãå¦æå®è£è¿ç¨éå°é®é¢ï¼ä¹å¯åèå®ä»¬çå®è£è¯´æææ¡£ã



> **Warning** 
>
> å¦æçµèä¸­ä»æªå®è£è¿ `PyTorch`ï¼`OpenCV` pythonåï¼åæ¬¡å®è£å¯è½ä¼éå°ä¸å°é®é¢ï¼ä½ä¸è¬é½æ¯å¸¸è§é®é¢ï¼å¯ä»¥èªè¡ç¾åº¦/Googleè§£å³ã



## æ¥å£è¯´æ

### ç±»åå§å

ä¸»ç±»ä¸º [**Pix2Text**](pix2text/pix_to_text.py) ï¼å¶åå§åå½æ°å¦ä¸ï¼

```python
class Pix2Text(object):
    def __init__(
        self,
        *,
        clf_config: Dict[str, Any] = None,
        general_config: Dict[str, Any] = None,
        english_config: Dict[str, Any] = None,
        formula_config: Dict[str, Any] = None,
        thresholds: Dict[str, Any] = None,
        device: str = 'cpu',  # ['cpu', 'cuda', 'gpu']
        **kwargs,
    ):
```

å¶ä¸­çååæ°è¯´æå¦ä¸ï¼
* `clf_config` (dict): åç±»æ¨¡åå¯¹åºçéç½®ä¿¡æ¯ï¼é»è®¤ä¸º `None`ï¼è¡¨ç¤ºä½¿ç¨é»è®¤éç½®ï¼
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
        'model_fp': None  # å¦ææå®ï¼ç´æ¥ä½¿ç¨æ­¤æ¨¡åæä»¶
  }
	```
	
* `general_config` (dict): éç¨æ¨¡åå¯¹åºçéç½®ä¿¡æ¯ï¼é»è®¤ä¸º `None`ï¼è¡¨ç¤ºä½¿ç¨é»è®¤éç½®ï¼

  ```python
  {}
  ```

* `english_config` (dict): è±ææ¨¡åå¯¹åºçéç½®ä¿¡æ¯ï¼é»è®¤ä¸º `None`ï¼è¡¨ç¤ºä½¿ç¨é»è®¤éç½®ï¼

  ```py
  {'det_model_name': 'en_PP-OCRv3_det', 'rec_model_name': 'en_PP-OCRv3'}
  ```

* `formula_config` (dict): å¬å¼è¯å«æ¨¡åå¯¹åºçéç½®ä¿¡æ¯ï¼é»è®¤ä¸º `None`ï¼è¡¨ç¤ºä½¿ç¨é»è®¤éç½®ï¼

  ```python
  {
      'config': LATEX_CONFIG_FP,
      'checkpoint': Path(data_dir()) / 'formular' / 'weights.pth',
      'no_resize': False
  }
  ```

* `thresholds` (dict): è¯å«éå¼å¯¹åºçéç½®ä¿¡æ¯ï¼é»è®¤ä¸º `None`ï¼è¡¨ç¤ºä½¿ç¨é»è®¤éç½®ï¼

  ```py
  {
      'formula2general': 0.65,  # å¦æè¯å«ä¸º `formula` ç±»åï¼ä½å¾åå°äºæ­¤éå¼ï¼åæ¹ä¸º `general` ç±»å
      'english2general': 0.75,  # å¦æè¯å«ä¸º `english` ç±»åï¼ä½å¾åå°äºæ­¤éå¼ï¼åæ¹ä¸º `general` ç±»å
  }
  ```

* `device` (str): ä½¿ç¨ä»ä¹èµæºè¿è¡è®¡ç®ï¼æ¯æ `['cpu', 'cuda', 'gpu']`ï¼é»è®¤ä¸º `cpu`

* `**kwargs` (): é¢ççå¶ä»åæ°ï¼ç®åæªè¢«ä½¿ç¨



### è¯å«ç±»å½æ°

éè¿è°ç¨ç±» **`Pix2Text`** çç±»å½æ° `.recognize()` å®æå¯¹æå®å¾ççæå­æLatexè¯å«ãç±»å½æ° `.recognize()` è¯´æå¦ä¸ï¼

```py
    def recognize(self, img: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`

        Returns: a dict, with keys:
           `image_type`: å¾åç±»å«ï¼
           `text`: è¯å«åºçæå­æLatexå¬å¼

        """
```



å¶ä¸­çè¾å¥åæ°è¯´æå¦ä¸ï¼

* `img` (`str` or `Image.Image`)ï¼å¾è¯å«å¾ççè·¯å¾ï¼æèå©ç¨ `Image.open()` å·²è¯»å¥çå¾ç `Image` ã



è¿åç»æè¯´æå¦ä¸ï¼

* `image_type`ï¼è¯å«åºçå¾åç±»å«ï¼åå¼ä¸º `formula`ã`english` æè `general` ï¼
* `text`ï¼è¯å«åºçæå­æLatexå¬å¼ ã

å¦åé¢ç»åºçä¸ä¸ªç¤ºä¾ç»æï¼

```json
{"image_type": "general",
 "text": "618\nå¼é¨çº¢æåè´­\nå¾è´µ\nä¹°è´µè¿å·®\nç»äºéä»·äº\n"
          "100%æ¡èä¸\nè¦ä¹°è¶æ©\nä»æ¥ä¸å188å\nä»éä¸å¤©"}
```



`Pix2Text` ç±»ä¹å®ç°äº `__call__()` å½æ°ï¼å¶åè½ä¸ `.recognize()` å½æ°å®å¨ç¸åãæä»¥æä¼æä»¥ä¸çè°ç¨æ¹å¼ï¼

```python
from pix2text import Pix2Text

img_fp = './docs/examples/formula.jpg'
p2t = Pix2Text()
out_text = p2t(img_fp)  # ä¹å¯ä»¥ä½¿ç¨ `p2t.recognize(img_fp)` è·å¾ç¸åçç»æ
print(out_text)
```



## HTTPæå¡

 **Pix2Text** å å¥äºåºäº FastAPI çHTTPæå¡ãå¼å¯æå¡éè¦å®è£å ä¸ªé¢å¤çåï¼å¯ä»¥ä½¿ç¨ä»¥ä¸å½ä»¤å®è£ï¼

```bash
pip install pix2text[serve]
```



å®è£å®æåï¼å¯ä»¥éè¿ä»¥ä¸å½ä»¤å¯å¨HTTPæå¡ï¼**`-p`** åé¢çæ°å­æ¯**ç«¯å£**ï¼å¯ä»¥æ ¹æ®éè¦èªè¡è°æ´ï¼ï¼

```bash
p2t serve -p 8503
```



æå¡å¼å¯åï¼å¯ä»¥ä½¿ç¨ä»¥ä¸æ¹å¼è°ç¨æå¡ã



### å½ä»¤è¡

æ¯å¦å¾è¯å«æä»¶ä¸º `docs/examples/english.jpg`ï¼å¦ä¸ä½¿ç¨ `curl` è°ç¨æå¡ï¼

```bash
> curl -F image=@docs/examples/english.jpg http://0.0.0.0:8503/pix2text
```



### Python

ä½¿ç¨å¦ä¸æ¹å¼è°ç¨æå¡ï¼

```python
import requests

image_fp = 'docs/examples/english.jpg'
r = requests.post(
    'http://0.0.0.0:8503/pix2text', files={'image': (image_fp, open(image_fp, 'rb'), 'image/png')},
)
out = r.json()['results']
print(out)
```



### å¶ä»è¯­è¨

è¯·åç§ `curl` çè°ç¨æ¹å¼èªè¡å®ç°ã



## èæ¬è¿è¡

èæ¬ [scripts/screenshot_daemon.py](scripts/screenshot_daemon.py) å®ç°äºèªå¨å¯¹æªå±å¾çè°ç¨ Pixe2Text è¿è¡å¬å¼æèæå­è¯å«ãè¿ä¸ªåè½æ¯å¦ä½å®ç°çå¢ï¼



**ä»¥ä¸æ¯å·ä½çè¿è¡æµç¨ï¼è¯·åå®è£å¥½ Pix2Textï¼ï¼**

1. æ¾ä¸ä¸ªåæ¬¢çæªå±è½¯ä»¶ï¼è¿ä¸ªè½¯ä»¶åªè¦**æ¯æææªå±å¾çå­å¨å¨æå®æä»¶å¤¹**å³å¯ãæ¯å¦Macä¸åè´¹ç **Xnip** å°±å¾å¥½ç¨ã

2. é¤äºå®è£Pix2Textå¤ï¼è¿éè¦é¢å¤å®è£ä¸ä¸ªPythonå **pyperclip**ï¼å©ç¨å®æè¯å«ç»æå¤å¶è¿ç³»ç»çåªåæ¿ï¼

   ```bash
   $ pip install pyperclip
   ```

3. ä¸è½½èæ¬æä»¶ [scripts/screenshot_daemon.py](scripts/screenshot_daemon.py) å°æ¬å°ï¼ç¼è¾æ­¤æä»¶ `"SCREENSHOT_DIR"` æå¨è¡ï¼ç¬¬ `17` è¡ï¼ï¼æè·¯å¾æ¹ä¸ºä½ çæªå±å¾çæå­å¨çç®å½ã

4. è¿è¡æ­¤èæ¬ï¼

   ```bash
   $ python scripts/screenshot_daemon.py
   ```

å¥½äºï¼ç°å¨å°±ç¨ä½ çæªå±è½¯ä»¶è¯è¯ææå§ãæªå±åçè¯å«ç»æä¼åå¥å½åæä»¶å¤¹ç **`out-text.html`** æä»¶ï¼åªè¦å¨æµè§å¨ä¸­æå¼æ­¤æä»¶å³å¯çå°ææã



æ´è¯¦ç»ä½¿ç¨ä»ç»å¯åèè§é¢ï¼ã[Pix2Text: æ¿ä»£ Mathpix çåè´¹ Python å¼æºå·¥å·](https://www.bilibili.com/video/BV12e4y1871U)ãã

<div align="center">
  <img src="./docs/figs/html.jpg" alt="å¾®ä¿¡ç¾¤äºç»´ç " width="700px"/>
</div>

> **Note**
> 
> æè°¢æçåäºå¸®å¿å®æäºæ­¤é¡µé¢çå¤§é¨åå·¥ä½ãè¿ä¸ªé¡µé¢è¿æå¾å¤§æ¹è¿ç©ºé´ï¼æ¬¢è¿å¯¹åç«¯çæçæåå¸®å¿æPRä¼åæ­¤é¡µé¢ã



## ç»ä½èæ¥æ¯åå¡

å¼æºä¸æï¼å¦ææ­¤é¡¹ç®å¯¹æ¨æå¸®å©ï¼å¯ä»¥èè [ç»ä½èå ç¹æ²¹ð¥¤ï¼é¼é¼æ°ðªð»](https://dun.mianbaoduo.com/@breezedeus) ã

---

å®æ¹ä»£ç åºï¼[https://github.com/breezedeus/pix2text](https://github.com/breezedeus/pix2text)ã

