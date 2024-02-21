# coding: utf-8

from pix2text import Pix2Text, merge_line_texts
# from pix2text.latex_ocr import post_post_process_latex


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
        # 'model_name': 'mfr-pro',
        # 'model_backend': 'onnx',
    }
    p2t = Pix2Text(formula_config=formula_config)
    print(p2t.recognize(img_fp))
    # print(p2t.recognize_formula(img_fp))
    # outs = p2t(img_fp, resized_shape=608, save_analysis_res='./analysis_res.jpg')  # can also use `p2t.recognize(img_fp)`
    # print(outs)
    # # To get just the text contents, use:
    # only_text = merge_line_texts(outs, auto_line_break=True)
    # print(only_text)


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
