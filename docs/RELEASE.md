# Release Notes

## Update 2024.04.30：**V1.1.0.1** Released

Major changes:

* Fix the exception occurring when saving files on Windows.

主要变更： 

* 修复 Windows 下存储文件时出现的异常。


## Update 2024.04.28：**V1.1** Released

Major changes:

* Added layout analysis and table recognition models, supporting the conversion of images with complex layouts into Markdown format. See examples: [Pix2Text Online Documentation / Examples](https://pix2text.readthedocs.io/zh/latest/examples_en/).
* Added support for converting entire PDF files to Markdown format. See examples: [Pix2Text Online Documentation / Examples](https://pix2text.readthedocs.io/zh/latest/examples_en/).
* Enhanced the interface with more features, including adjustments to existing interface parameters.
* Launched the [Pix2Text Online Documentation](https://pix2text.readthedocs.io).

主要变更： 

* 加入了版面分析和表格识别模型，支持把复杂排版的图片转换为 Markdown 格式，示例见：[Pix2Text 在线文档/Examples](https://pix2text.readthedocs.io/zh/latest/examples/)。
* 支持把整个 PDF 文件转换为 Markdown 格式，示例见：[Pix2Text 在线文档/Examples](https://pix2text.readthedocs.io/zh/latest/examples/)。
* 加入了更丰富的接口，已有接口的参数也有所调整。
* 上线了 [Pix2Text 在线文档](https://pix2text.readthedocs.io)。
 

## Update 2024.03.30：**V1.0.2.3** Released

Major changes:

* Fixed the issue caused by `merge_line_texts`, see details at: https://github.com/breezedeus/Pix2Text/issues/84.
* Optimized the post-processing logic to handle some abnormal sequences.

主要变更： 

* 修复 `merge_line_texts` 带来的错误，具体见：https://github.com/breezedeus/Pix2Text/issues/84 。
* 优化了后处理逻辑，处理部分不正常的序列。

## Update 2024.03.18：**V1.0.2.2** Released

Major changes:

* The previously used `output_logits` argument is incompatible with transformers < 4.38.0, replaced by the `output_scores` argument. https://github.com/breezedeus/Pix2Text/issues/81
* Fixed a bug in `serve.py` that was not compatible with the new pix2text version.

主要变更： 

* 之前使用的 `output_logits` 参数不兼容 transformers < 4.38.0，换为 `output_scores` 参数。 https://github.com/breezedeus/Pix2Text/issues/81
* 修复 `serve.py` 中未兼容新版接口的 bug。

## Update 2024.03.15：**V1.0.2.1** Released

Major Changes:

* Fixed mishandling of LaTeX expressions during post-processing, such as replacing `\rightarrow` with `arrow`.
* Added `rec_config` parameter to `.recognize_text()` and `.recognize_formula()` methods for passing additional parameters for recognition.

主要变更：

* 修复对 LaTeX 表达式进行后处理时引入的误操作，如 `\rightarrow` 被替换为 `arrow`。
* 对 `.recognize_text()` 和 `.recognize_formula()` 加入了 `rec_config` 参数，以便传入用于识别的额外参数。

## Update 2024.03.14：**V1.0.2** Released

Major Changes:

* Optimized the recognition process, improving the recognition of boundary punctuation that may have been missed before.
* Enhanced the LaTeX recognition results by restoring the formula tags to the formulas.
* Adjusted the output format of the recognition results, adding the `return_text` parameter to control whether to return only text or more detailed information. When returning more detailed information, confidence score `score` and position information `position` will also be provided. Thanks to [@hiroi-sora](https://github.com/hiroi-sora) for the suggestion: https://github.com/breezedeus/Pix2Text/issues/67.

主要变更：

* 优化了识别的逻辑，以前可能漏识的边界标点现在可以比较好的识别。
* 对 Latex 识别结果进行了优化，把公式的 tag 还原到公式中。
* 调整了识别结果的输出格式，增加了参数 `return_text` 来控制结果是只返回文本还是更丰富的信息。当返回更丰富信息时，会返回置信度 `score` 以及位置信息 `position`。感谢 [@hiroi-sora](https://github.com/hiroi-sora) 的建议：https://github.com/breezedeus/Pix2Text/issues/67 。

## Update 2024.03.03：发布 **V1.0.1**

主要变更：

* 修复在 CUDA 环境下使用 `LatexOCR` 时出现的错误，具体见：https://github.com/breezedeus/Pix2Text/issues/65#issuecomment-1973037910 ，感谢 [@MSZ-006NOC](https://github.com/MSZ-006NOC)。


## Update 2024.02.26：发布 **V1.0**

主要变更：

* 数学公式识别（MFR）模型使用新架构，在新的数据集上训练，获得了 SOTA 的精度。具体说明请见：[Pix2Text V1.0 新版发布：最好的开源公式识别模型 | Breezedeus.com](https://www.breezedeus.com/article/p2t-v1.0)。


## Update 2024.01.10：发布 **V0.3**

主要变更：

* 支持识别 **`80+` 种语言**，详细语言列表见 [支持的语言列表](./README_cn.md#支持的语言列表)；

* 模型自动下载增加国内站点；

* 优化对检测 boxes 的合并逻辑。

  

## Update 2023.12.21：发布 **V0.2.3.3**

主要变更：

* fix: bugfixed from [@hiroi-sora](https://github.com/hiroi-sora) , thanks much.

  

## Update 2023.09.10：发布 **V0.2.3.2**

主要变更：
* fix: 去掉 `consts.py` 无用的 `CATEGORY_MAPPINGS`。

## Update 2023.07.14：发布 **V0.2.3.1**

主要变更：
* 修复了 `self.recognize_by_clf` 返回结果中不包含 `line_number` 字段导致 `merge_line_texts` 报错的bug。

## Update 2023.07.03：发布 **V0.2.3**

主要变更：
* 优化了对检测出的boxes的排序逻辑，以及对混合图片的处理逻辑，使得最终识别效果更符合直觉。具体参考：[Pix2Text 新版公式识别模型 | Breezedeus.com](https://www.breezedeus.com/article/p2t-mfd-20230702) 。
* 修复了模型文件自动下载的功能。HuggingFace似乎对下载文件的逻辑做了调整，导致之前版本的自动下载失败，当前版本已修复。但由于HuggingFace国内被墙，国内下载仍需 **梯子（VPN）**。
* 更新了各个依赖包的版本号。


## Update 2023.06.20：发布新版 MFD 模型

主要变更：
* 基于新标注的数据，重新训练了 **MFD YoloV7** 模型，目前新模型已部署到 [P2T网页版](https://p2t.breezedeus.com) 。具体说明见：[Pix2Text (P2T) 新版公式检测模型 | Breezedeus.com](https://www.breezedeus.com/article/p2t-mfd-20230613) 。
* 之前的 MFD YoloV7 模型已开放给星球会员下载，具体说明见：[P2T YoloV7 数学公式检测模型开放给星球会员下载 | Breezedeus.com](https://www.breezedeus.com/article/p2t-yolov7-for-zsxq-20230619) 。


## Update 2023.02.19：发布 **V0.2.2.1**

主要变更：
* 修复bug。


## Update 2023.02.19：发布 **V0.2.2**

主要变更：
* 修复旋转框导致的识别结果错误；
* 去掉代码中不小心包含的 `breakpoint()`。


## [Yanked] Update 2023.02.19：发布 **V0.2.1**

主要变更：
* 增加后处理机制优化Latex-OCR的识别结果；
* 使用最新的 [CnSTD](https://github.com/breezedeus/cnstd) 和 [CnOCR](https://github.com/breezedeus/cnocr)，它们修复了一些bug。

## Update 2023.02.03：发布 **V0.2**

主要变更：
* 利用 **[CnSTD](https://github.com/breezedeus/cnstd)** 新版的**数学公式检测**（**Mathematical Formula Detection**，简称 **MFD**）能力，**P2T V0.2** 支持**识别既包含文字又包含公式的混合图片**。

## Update 2022.10.21：发布 V0.1.1

主要变更：
* Fix: remove the character which causes error on Windows

## Update 2022.09.11：发布 V0.1
* 初版发布
