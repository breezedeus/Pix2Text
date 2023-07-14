# Release Notes

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
* 基于新标注的数据，重新训练了 **MFD YoloV7** 模型，目前新模型已部署到 [P2T网页版](https://p2t.behye.com) 。具体说明见：[Pix2Text (P2T) 新版公式检测模型 | Breezedeus.com](https://www.breezedeus.com/article/p2t-mfd-20230613) 。
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
