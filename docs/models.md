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

> 注意，Pix2Text 的付费模型包含不同的 license 版本，购买时请参考具体的产品说明。

下面的说明主要针对免费的基础模型。

## 版面分析模型
**版面分析模型** 下载地址：[breezedeus/pix2text-layout](https://huggingface.co/breezedeus/pix2text-layout) （不能科学上网请使用 [国内地址](https://hf-mirror.com/breezedeus/pix2text-layout)）。
把这里面的所有文件都下载到 `~/.pix2text/1.1/layout-parser` （Windows 系统放在 `C:\Users\<username>\AppData\Roaming\pix2text\1.1\layout-parser`）目录下即可，目录不存在的话请自己创建。

> 注：上面路径的 `1.1` 是 pix2text 的版本号，`1.1.*` 都对应 `1.1`。如果是其他版本请自行替换。

## 表格识别模型
**表格识别模型** 下载地址：[breezedeus/pix2text-table-rec](https://huggingface.co/breezedeus/pix2text-table-rec) （不能科学上网请使用 [国内地址](https://hf-mirror.com/breezedeus/pix2text-table-rec)）。
把这里面的所有文件都下载到 `~/.pix2text/1.1/table-rec` （Windows 系统放在 `C:\Users\<username>\AppData\Roaming\pix2text\1.1\table-rec`）目录下即可，目录不存在的话请自己创建。

> 注：上面路径的 `1.1` 是 pix2text 的版本号，`1.1.*` 都对应 `1.1`。如果是其他版本请自行替换。

## 文字识别引擎
CnOCR 模型下载请参考 [CnOCR在线文档/可用模型](https://cnocr.readthedocs.io/zh/latest/models/)。


EasyOCR 模型下载请参考 [EasyOCR](https://github.com/JaidedAI/EasyOCR)。

## 数学公式检测模型
**数学公式检测模型**（MFD）来自 [CnSTD](https://github.com/breezedeus/cnstd) 的数学公式检测模型（MFD），请参考其代码库说明。


## 数学公式识别模型
**数学公式识别模型** 下载地址：[breezedeus/pix2text-mfr](https://huggingface.co/breezedeus/pix2text-mfr) （不能科学上网请使用 [国内地址](https://hf-mirror.com/breezedeus/pix2text-mfr)）。
把这里面的所有文件都下载到 `~/.pix2text/1.1/mfr-onnx` （Windows 系统放在 `C:\Users\<username>\AppData\Roaming\pix2text\1.1\mfr-onnx`）目录下即可，目录不存在的话请自己创建。

> 注：上面路径的 `1.1` 是 pix2text 的版本号，`1.1.*` 都对应 `1.1`。如果是其他版本请自行替换。
