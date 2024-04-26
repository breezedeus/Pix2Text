# 各种模型

**Pix2Text (P2T)** 整合了很多不同功能的模型，主要包括：

- **版面分析模型**：[breezedeus/pix2text-layout](https://huggingface.co/breezedeus/pix2text-layout) （[国内地址](https://hf-mirror.com/breezedeus/pix2text-layout)）。
- **表格识别模型**：[breezedeus/pix2text-table-rec](https://huggingface.co/breezedeus/pix2text-table-rec) （[国内地址](https://hf-mirror.com/breezedeus/pix2text-table-rec)）。
- **文字识别引擎**：支持 **`80+` 种语言**，如**英文、简体中文、繁体中文、越南语**等。其中，**英文**和**简体中文**识别使用的是开源 OCR 工具 [CnOCR](https://github.com/breezedeus/cnocr) ，其他语言的识别使用的是开源 OCR 工具 [EasyOCR](https://github.com/JaidedAI/EasyOCR) 。
- **数学公式检测模型（MFD）**：来自 [CnSTD](https://github.com/breezedeus/cnstd) 的数学公式检测模型（MFD）。
- **数学公式识别模型（MFR）**：[breezedeus/pix2text-mfr](https://huggingface.co/breezedeus/pix2text-mfr) （[国内地址](https://hf-mirror.com/breezedeus/pix2text-mfr)）。

其中多个模型来自其他开源作者， 非常感谢他们的贡献。

这些模型正常情况下都会自动下载（可能会比较慢，只要不报错请勿手动打断下载过程），但如果下载失败，可以参考以下的说明手动下载。

除基础模型外，Pix2Text 还提供了以下模型的高级付费版：

- MFD 和 MFR 付费模型：具体参考 [P2T详细资料 | Breezedeus.com](https://www.breezedeus.com/article/pix2text_cn)。
- CnOCR 付费模型：具体参考 [CnOCR详细资料 | Breezedeus.com](https://www.breezedeus.com/article/cnocr)。

具体说明请见本页面末尾。

下面的说明主要针对免费的基础模型。

## 版面分析模型
**版面分析模型** 下载地址：[breezedeus/pix2text-layout](https://huggingface.co/breezedeus/pix2text-layout) （不能科学上网请使用 [国内地址](https://hf-mirror.com/breezedeus/pix2text-layout)）。
把这里面的所有文件都下载到 `~/.pix2text/1.1/layout-parser` （Windows 系统放在 `C:\Users\<username>\AppData\Roaming\pix2text\1.1\layout-parser`）目录下即可，目录不存在的话请自己创建。

> 注：上面路径的 `1.1` 是 pix2text 的版本号，`1.1.*` 都对应 `1.1`。如果是其他版本请自行替换。

## 表格识别模型
**表格识别模型** 下载地址：[breezedeus/pix2text-table-rec](https://huggingface.co/breezedeus/pix2text-table-rec) （不能科学上网请使用 [国内地址](https://hf-mirror.com/breezedeus/pix2text-table-rec)）。
把这里面的所有文件都下载到 `~/.pix2text/1.1/table-rec` （Windows 系统放在 `C:\Users\<username>\AppData\Roaming\pix2text\1.1\table-rec`）目录下即可，目录不存在的话请自己创建。

> 注：上面路径的 `1.1` 是 pix2text 的版本号，`1.1.*` 都对应 `1.1`。如果是其他版本请自行替换。

## 数学公式检测模型
**数学公式检测模型**（MFD）来自 [CnSTD](https://github.com/breezedeus/cnstd) 的数学公式检测模型（MFD），请参考其代码库说明。

如果系统无法自动成功下载模型文件，则需要手动从 [**cnstd-cnocr-models**](https://huggingface.co/breezedeus/cnstd-cnocr-models) （[国内镜像](https://hf-mirror.com/breezedeus/cnstd-cnocr-models)）项目中下载，或者从[百度云盘](https://pan.baidu.com/s/1zDMzArCDrrXHWL0AWxwYQQ?pwd=nstd)（提取码为 `nstd`）下载对应的zip文件并把它存放于 `~/.cnstd/1.2`（Windows下为 `C:\Users\<username>\AppData\Roaming\cnstd\1.2`）目录中。

## 数学公式识别模型
**数学公式识别模型** 下载地址：[breezedeus/pix2text-mfr](https://huggingface.co/breezedeus/pix2text-mfr) （不能科学上网请使用 [国内地址](https://hf-mirror.com/breezedeus/pix2text-mfr)）。
把这里面的所有文件都下载到 `~/.pix2text/1.1/mfr-onnx` （Windows 系统放在 `C:\Users\<username>\AppData\Roaming\pix2text\1.1\mfr-onnx`）目录下即可，目录不存在的话请自己创建。

> 注：上面路径的 `1.1` 是 pix2text 的版本号，`1.1.*` 都对应 `1.1`。如果是其他版本请自行替换。

## 文字识别引擎
Pix2Text 的**文字识别引擎**可以识别 **`80+` 种语言**，如**英文、简体中文、繁体中文、越南语**等。其中，**英文**和**简体中文**识别使用的是开源 OCR 工具 [CnOCR](https://github.com/breezedeus/cnocr) ，其他语言的识别使用的是开源 OCR 工具 [EasyOCR](https://github.com/JaidedAI/EasyOCR) 。

正常情况下，CnOCR 的模型都会自动下载。如果无法自动下载，可以参考以下说明手动下载。
CnOCR 的开源模型都放在 [**cnstd-cnocr-models**](https://huggingface.co/breezedeus/cnstd-cnocr-models) （[国内镜像](https://hf-mirror.com/breezedeus/cnstd-cnocr-models)）项目中，可免费下载使用。
如果下载太慢，也可以从 [百度云盘](https://pan.baidu.com/s/1RhLBf8DcLnLuGLPrp89hUg?pwd=nocr) 下载， 提取码为 `nocr`。具体方法可参考 [CnOCR在线文档/使用方法](https://cnocr.readthedocs.io/zh/latest/usage) 。

CnOCR 中的文字检测引擎使用的是 [CnSTD](https://github.com/breezedeus/cnstd)，
如果系统无法自动成功下载模型文件，则需要手动从 [**cnstd-cnocr-models**](https://huggingface.co/breezedeus/cnstd-cnocr-models) （[国内镜像](https://hf-mirror.com/breezedeus/cnstd-cnocr-models)）项目中下载，或者从[百度云盘](https://pan.baidu.com/s/1zDMzArCDrrXHWL0AWxwYQQ?pwd=nstd)（提取码为 `nstd`）下载对应的zip文件并把它存放于 `~/.cnstd/1.2`（Windows下为 `C:\Users\<username>\AppData\Roaming\cnstd\1.2`）目录中。

关于 CnOCR 模型的更多信息请参考 [CnOCR在线文档/可用模型](https://cnocr.readthedocs.io/zh/latest/models)。

CnOCR 也提供**高级版的付费模型**，具体参考本文末尾的说明。

- CnOCR 付费模型：具体参考 [CnOCR详细资料 | Breezedeus.com](https://www.breezedeus.com/article/cnocr)。

<br/>

EasyOCR 模型下载请参考 [EasyOCR](https://github.com/JaidedAI/EasyOCR)。

## 高级版付费模型

除基础模型外，Pix2Text 还提供了以下模型的高级付费版：

- MFD 和 MFR 付费模型：具体参考 [P2T详细资料 | Breezedeus.com](https://www.breezedeus.com/article/pix2text_cn)。
- CnOCR 付费模型：具体参考 [CnOCR详细资料 | Breezedeus.com](https://www.breezedeus.com/article/cnocr)。

> 注意，付费模型包含不同的 license 版本，购买时请参考具体的产品说明。

建议购买前首先使用 **[在线 Demo](https://huggingface.co/spaces/breezedeus/Pix2Text-Demo)**（无法科学上网可以使用 [国内 Demo](https://hf-mirror.com/spaces/breezedeus/Pix2Text-Demo)）**验证模型效果后再购买**。

**模型购买地址**：

| 模型名称         | 购买地址                                          | 说明 
|--------------|-----------------------------------------------|-----------------------------------------------------------------------------------|
| MFD pro 模型   | [Lemon Squeezy](https://ocr.lemonsqueezy.com) | 包含企业版和个人版，可开发票。具体说明见：[P2T详细资料](https://www.breezedeus.com/article/pix2text_cn)    | 
| MFD pro 模型   | [B站工房](https://gf.bilibili.com/item/detail/1102870055)          | 仅包含个人版，不可商用，不能开发票。具体说明见：[P2T详细资料](https://www.breezedeus.com/article/pix2text_cn) | 
| MFR pro 模型   | [Lemon Squeezy](https://ocr.lemonsqueezy.com) | 包含企业版和个人版，可开发票。具体说明见：[P2T详细资料](https://www.breezedeus.com/article/pix2text_cn)    | 
| MFR pro 模型   | [B站工房](https://gf.bilibili.com/item/detail/1103052055)          | 仅包含个人版，不可商用，不能开发票。具体说明见：[P2T详细资料](https://www.breezedeus.com/article/pix2text_cn) | 
| CnOCR pro 模型 | [Lemon Squeezy](https://ocr.lemonsqueezy.com) | 包含企业版和个人版，可开发票。具体说明见：[P2T详细资料](https://www.breezedeus.com/article/pix2text_cn) 和 [CnOCR详细资料](https://www.breezedeus.com/article/cnocr) | 
| CnOCR pro 模型 | [B站工房](https://gf.bilibili.com/item/detail/1104820055) | 仅包含个人版，不可商用，不能开发票。具体说明见：[P2T详细资料](https://www.breezedeus.com/article/pix2text_cn) 和 [CnOCR详细资料](https://www.breezedeus.com/article/cnocr) | 

购买过程遇到问题可以扫码加小助手为好友进行沟通，备注 `p2t`，小助手会尽快答复：

<figure markdown>
![微信交流群](https://huggingface.co/datasets/breezedeus/cnocr-wx-qr-code/resolve/main/wx-qr-code.JPG){: style="width:270px"}
</figure>

更多联系方式见 [交流群](contact.md)。