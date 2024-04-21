# 安装

## pip 安装

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
 

## GPU 环境使用 ONNX 模型

默认情况下安装的 **ONNX** 包是 **`onnxruntime`**，它只能在 `CPU` 上运行。如果需要在 `GPU` 环境使用 **ONNX** 模型，需要卸载此包，然后安装包 **`onnxruntime-gpu`** 。

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```
