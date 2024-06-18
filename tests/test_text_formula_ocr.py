# coding: utf-8

import os

from pix2text import TextFormulaOCR, merge_line_texts


def test_mfd():
    config = dict()
    model = TextFormulaOCR.from_config(config)

    res = model.recognize(
        './docs/examples/zh1.jpg', save_analysis_res='./analysis_res.jpg',
    )
    print(res)


def test_example():
    # img_fp = './docs/examples/formula.jpg'
    img_fp = './docs/examples/mixed.jpg'
    formula_config = {
        'model_name': 'mfr-pro',
        'model_backend': 'onnx',
    }
    p2t = TextFormulaOCR.from_config(total_configs={'formula': formula_config})
    print(p2t.recognize(img_fp))
    # print(p2t.recognize_formula(img_fp))
    # outs = p2t(img_fp, resized_shape=608, save_analysis_res='./analysis_res.jpg')  # can also use `p2t.recognize(img_fp)`
    # print(outs)
    # # To get just the text contents, use:
    # only_text = merge_line_texts(outs, auto_line_break=True)
    # print(only_text)


def test_blog_example():
    img_fp = './docs/examples/mixed.jpg'

    total_config = dict(
        mfd=dict(  # 声明 MFD 的初始化参数
            model_path=os.path.expanduser(
                '~/.pix2text/1.1/mfd-onnx/mfd-v20240618.onnx'
            ),  # 注：修改成你的模型文件所存储的路径
        ),
        formula=dict(
            model_name='mfr-pro',
            model_backend='onnx',
            model_dir=os.path.expanduser(
                '~/.pix2text/1.1/mfr-pro-onnx'
            ),  # 注：修改成你的模型文件所存储的路径
        ),
    )
    p2t = TextFormulaOCR.from_config(total_configs=total_config)
    outs = p2t.recognize(
        img_fp, resized_shape=608, return_text=False
    )  # 也可以使用 `p2t(img_fp)` 获得相同的结果
    print(outs)
    # 如果只需要识别出的文字和Latex表示，可以使用下面行的代码合并所有结果
    only_text = merge_line_texts(outs, auto_line_break=True)
    print(only_text)


def test_blog_pro_example():
    img_fp = './docs/examples/mixed.jpg'

    total_config = dict(
        languages=('en', 'ch_sim'),
        mfd=dict(  # 声明 MFD 的初始化参数
            model_path=os.path.expanduser(
                '~/.pix2text/1.1/mfd-onnx/mfd-v20240618.onnx'
            ),  # 注：修改成你的模型文件所存储的路径
        ),
        formula=dict(
            model_name='mfr-pro',
            model_backend='onnx',
            model_dir=os.path.expanduser(
                '~/.pix2text/1.1/mfr-pro-onnx'
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
    p2t = TextFormulaOCR.from_config(total_configs=total_config)
    outs = p2t.recognize(
        img_fp, resized_shape=608, return_text=False
    )  # 也可以使用 `p2t(img_fp)` 获得相同的结果
    print(outs)
    # 如果只需要识别出的文字和Latex表示，可以使用下面行的代码合并所有结果
    only_text = merge_line_texts(outs, auto_line_break=True)
    print(only_text)


def test_example_mixed():
    img_fp = './docs/examples/en1.jpg'
    p2t = TextFormulaOCR.from_config()
    outs = p2t.recognize(
        img_fp, resized_shape=608, return_text=False
    )  # 也可以使用 `p2t(img_fp)` 获得相同的结果
    print(outs)
    # 如果只需要识别出的文字和Latex表示，可以使用下面行的代码合并所有结果
    only_text = merge_line_texts(outs, auto_line_break=True)
    print(only_text)


def test_example_formula():
    img_fp = './docs/examples/math-formula-42.png'
    p2t = TextFormulaOCR.from_config()
    outs = p2t.recognize_formula(img_fp)
    print(outs)


def test_example_text():
    img_fp = './docs/examples/general.jpg'
    p2t = TextFormulaOCR()
    outs = p2t.recognize_text(img_fp)
    print(outs)
