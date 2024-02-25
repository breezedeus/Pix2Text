# coding: utf-8

import os

from pix2text import Pix2Text, merge_line_texts


def test_mfd():
    config = dict(analyzer=dict(model_name='mfd'))
    p2t = Pix2Text.from_config(config)

    res = p2t.recognize(
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
    p2t = Pix2Text(formula_config=formula_config)
    print(p2t.recognize(img_fp))
    # print(p2t.recognize_formula(img_fp))
    # outs = p2t(img_fp, resized_shape=608, save_analysis_res='./analysis_res.jpg')  # can also use `p2t.recognize(img_fp)`
    # print(outs)
    # # To get just the text contents, use:
    # only_text = merge_line_texts(outs, auto_line_break=True)
    # print(only_text)


def test_blog_example():
    img_fp = './docs/examples/mixed.jpg'

    p2t = Pix2Text(
        analyzer_config=dict(  # 声明 LayoutAnalyzer 的初始化参数
            model_name='mfd',
            model_type='yolov7',  # 表示使用的是 YoloV7 模型，而不是 YoloV7_Tiny 模型
            model_fp=os.path.expanduser(
                '~/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt'
            ),  # 注：修改成你的模型文件所存储的路径
        ),
        formula_config=dict(
            model_name='mfr-pro',
            model_backend='onnx',
            model_dir=os.path.expanduser(
                '~/.pix2text/1.0/mfr-pro-onnx'
            ),  # 注：修改成你的模型文件所存储的路径
        ),
    )
    outs = p2t.recognize(img_fp, resized_shape=608)  # 也可以使用 `p2t(img_fp)` 获得相同的结果
    print(outs)
    # 如果只需要识别出的文字和Latex表示，可以使用下面行的代码合并所有结果
    only_text = merge_line_texts(outs, auto_line_break=True)
    print(only_text)


def test_blog_pro_example():
    img_fp = './docs/examples/mixed.jpg'

    p2t = Pix2Text(
        languages=('en', 'ch_sim'),
        analyzer_config=dict(  # 声明 LayoutAnalyzer 的初始化参数
            model_name='mfd',
            model_type='yolov7',  # 表示使用的是 YoloV7 模型，而不是 YoloV7_Tiny 模型
            model_fp=os.path.expanduser(
                '~/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt'
            ),  # 注：修改成你的模型文件所存储的路径
        ),
        formula_config=dict(
            model_name='mfr-pro',
            model_backend='onnx',
            model_dir=os.path.expanduser(
                '~/.pix2text/1.0/mfr-pro-onnx'
            ),  # 注：修改成你的模型文件所存储的路径
        ),
        text_config=dict(
            rec_model_name='doc-densenet_lite_666-gru_large',
            rec_model_backend='onnx',
            rec_model_fp=os.path.expanduser(
                '~/.cnocr/2.3/doc-densenet_lite_666-gru_large/cnocr-v2.3-doc-densenet_lite_666-gru_large-epoch=005-ft-model.onnx'  # noqa
            ),  # 注：修改成你的模型文件所存储的路径
        ),
    )
    outs = p2t.recognize(img_fp, resized_shape=608)  # 也可以使用 `p2t(img_fp)` 获得相同的结果
    print(outs)
    # 如果只需要识别出的文字和Latex表示，可以使用下面行的代码合并所有结果
    only_text = merge_line_texts(outs, auto_line_break=True)
    print(only_text)


def test_example_mixed():
    img_fp = './docs/examples/en1.jpg'
    p2t = Pix2Text()
    outs = p2t.recognize(img_fp, resized_shape=608)  # 也可以使用 `p2t(img_fp)` 获得相同的结果
    print(outs)
    # 如果只需要识别出的文字和Latex表示，可以使用下面行的代码合并所有结果
    only_text = merge_line_texts(outs, auto_line_break=True)
    print(only_text)


def test_example_formula():
    from pix2text import Pix2Text

    img_fp = './docs/examples/math-formula-42.png'
    p2t = Pix2Text()
    outs = p2t.recognize_formula(img_fp)
    print(outs)


def test_example_text():
    from pix2text import Pix2Text

    img_fp = './docs/examples/general.jpg'
    p2t = Pix2Text()
    outs = p2t.recognize_text(img_fp)
    print(outs)
