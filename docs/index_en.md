<figure markdown>
![Pix2Text](figs/p2t-logo.png){: style="width:180px"}
</figure>

# Pix2Text (P2T)
[![Discord](https://img.shields.io/discord/1200765964434821260?label=Discord)](https://discord.gg/GgD87WM8Tf)
[![Downloads](https://static.pepy.tech/personalized-badge/pix2text?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/pix2text)
[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fpix2text.readthedocs.io%2Fzh%2Flatest%2F&label=Visitors&countColor=%23f5c791&style=flat&labelStyle=none)](https://visitorbadge.io/status?path=https%3A%2F%2Fpix2text.readthedocs.io%2Fzh%2Flatest%2F)
[![license](https://img.shields.io/github/license/breezedeus/pix2text)](./LICENSE)
[![PyPI version](https://badge.fury.io/py/pix2text.svg)](https://badge.fury.io/py/pix2text)
[![forks](https://img.shields.io/github/forks/breezedeus/pix2text)](https://github.com/breezedeus/pix2text)
[![stars](https://img.shields.io/github/stars/breezedeus/pix2text)](https://github.com/breezedeus/pix2text)
![last-release](https://img.shields.io/github/release-date/breezedeus/pix2text)
![last-commit](https://img.shields.io/github/last-commit/breezedeus/pix2text)
[![Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fbreezedeus)](https://twitter.com/breezedeus)

<figure markdown>
[📖 Usage](usage.md) |
[🛠️ Installation](install.md) |
[🧳 Models](models.md) |
[🛀🏻 Demo](demo.md) |
[💬 Contact](contact.md)

[English](https://github.com/breezedeus/pix2text/blob/master/README.md) | 中文
</figure>

**Pix2Text (P2T)** aims to be a **free and open-source Python** alternative to **[Mathpix](https://mathpix.com/)**, and it can already accomplish **Mathpix**'s core functionality. **Pix2Text (P2T) can recognize layouts, tables, images, text, mathematical formulas, and integrate all of these contents into Markdown format. P2T can also convert an entire PDF file (which can contain scanned images or any other format) into Markdown format.**

**Pix2Text (P2T)** integrates the following models:

- **Layout Analysis Model**: [breezedeus/pix2text-layout](https://huggingface.co/breezedeus/pix2text-layout) ([Mirror](https://hf-mirror.com/breezedeus/pix2text-layout)).
- **Table Recognition Model**: [breezedeus/pix2text-table-rec](https://huggingface.co/breezedeus/pix2text-table-rec) ([Mirror](https://hf-mirror.com/breezedeus/pix2text-table-rec)).
- **Text Recognition Engine**: Supports **80+ languages** such as **English, Simplified Chinese, Traditional Chinese, Vietnamese**, etc. For English and Simplified Chinese recognition, it uses the open-source OCR tool [CnOCR](https://github.com/breezedeus/cnocr), while for other languages, it uses the open-source OCR tool [EasyOCR](https://github.com/JaidedAI/EasyOCR).
- **Mathematical Formula Detection Model (MFD)**: Mathematical formula detection model (MFD) from [CnSTD](https://github.com/breezedeus/cnstd).
- **Mathematical Formula Recognition Model (MFR)**: [breezedeus/pix2text-mfr](https://huggingface.co/breezedeus/pix2text-mfr) ([Mirror](https://hf-mirror.com/breezedeus/pix2text-mfr)).

Several models are contributed by other open-source authors, and their contributions are highly appreciated. 

For detailed explanations, please refer to the [Models](models.md).

As a Python3 toolkit, P2T may not be very user-friendly for those who are not familiar with Python. Therefore, we also provide a **[free-to-use P2T Online Web](https://p2t.breezedeus.com)**, where you can directly upload images and get P2T parsing results. The web version uses the latest models, resulting in better performance compared to the open-source models.

If you're interested, feel free to add the assistant as a friend by scanning the QR code and mentioning `p2t`. The assistant will regularly invite everyone to join the group where the latest developments related to P2T tools will be announced:

<div align="center">
  <img src="figs/wx-qr-code.JPG" alt="微信群二维码" width="300px"/>
</div>

The author also maintains a **Knowledge Planet** [**P2T/CnOCR/CnSTD Private Group**](https://t.zsxq.com/FEYZRJQ), where questions are answered promptly. You're welcome to join. The **knowledge planet private group** will also gradually release some private materials related to P2T/CnOCR/CnSTD, including **some unreleased models**, **discounts on purchasing premium models**, **code snippets for different application scenarios**, and answers to difficult problems encountered during use. The planet will also publish the latest research materials related to P2T/OCR/STD.

For more information, please refer to [Contact](contact.md).


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

也可以使用 **[在线 Demo](https://huggingface.co/spaces/breezedeus/Pix2Text-Demo)**（无法科学上网可以使用 [在线 Demo](https://hf-mirror.com/spaces/breezedeus/Pix2Text-Demo)） 尝试 **P2T** 在不同语言上的效果。但在线 Demo 使用的硬件配置较低，速度会较慢。如果是简体中文或者英文图片，建议使用 **[P2T网页版](https://p2t.breezedeus.com)**。


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

更多说明参考 [安装说明](install.md) 。

## HTTP 服务

使用命令 **`p2t serve`** 开启一个 HTTP 服务，用于接收图片（当前不支持 PDF）并返回识别结果。

```bash
p2t serve -l en,ch_sim -H 0.0.0.0 -p 8503
```

之后可以使用 curl 调用服务：

```bash
curl -X POST \
  -F "file_type=page" \
  -F "resized_shape=768" \
  -F "embed_sep= $,$ " \
  -F "isolated_sep=$$\n, \n$$" \
  -F "image=@docs/examples/page2.png;type=image/jpeg" \
  http://0.0.0.0:8503/pix2text
```

更多说明参考 [命令说明/开启服务](command.md) 。

## Mac 桌面客户端

请参考 [Pix2Text-Mac](https://github.com/breezedeus/Pix2Text-Mac) 安装 Pix2Text 的 MacOS 桌面客户端。

<div align="center">
  <img src="https://github.com/breezedeus/Pix2Text-Mac/raw/main/assets/on_menu_bar.jpg" alt="Pix2Text Mac 客户端" width="400px"/>
</div>


## 给作者来杯咖啡

开源不易，如果此项目对您有帮助，可以考虑 [给作者加点油🥤，鼓鼓气💪🏻](buymeacoffee.md) 。

---

官方代码库：

* **Github**: [https://github.com/breezedeus/pix2text](https://github.com/breezedeus/pix2text) 。
* **Gitee**: [https://gitee.com/breezedeus/pix2text](https://gitee.com/breezedeus/pix2text) 。

Pix2Text (P2T) 更多信息：[https://www.breezedeus.com/article/pix2text_cn](https://www.breezedeus.com/article/pix2text_cn) 。
