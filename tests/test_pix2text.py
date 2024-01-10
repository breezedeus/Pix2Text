# coding: utf-8

from pix2text import Pix2Text, merge_line_texts
from pix2text.latex_ocr import post_post_process_latex


def test_mfd():
    config = dict(analyzer=dict(model_name='mfd'))
    p2t = Pix2Text.from_config(config)

    res = p2t.recognize(
        './docs/examples/zh1.jpg', save_analysis_res='./analysis_res.jpg',
    )
    print(res)


def test_example():
    img_fp = './docs/examples/formula.jpg'
    p2t = Pix2Text()
    outs = p2t(img_fp, resized_shape=608)  # # can also use `p2t.recognize(img_fp)`
    print(outs)
    # To get just the text contents, use:
    only_text = merge_line_texts(outs, auto_line_break=True)
    print(only_text)


def test_post_post_process():
    latex = r'\log(p(x\mid q(x)))+||\mathrm{sg}\left[z_{e}(x)\right]-e||_{2}^{2}+\beta\left||z_{e}(x)-\mathrm{sg}[e]|_{2}^{2}'  # noqa: E501
    out = post_post_process_latex(latex)
    print(f'{latex}\n->\n{out}\n')
    # 以下是其他的测试字符串来测试此接口功能是否符合预期
    latex = r'(a+b)^2 +（a-b）^2'
    out = post_post_process_latex(latex)
    print(f'{latex}\n->\n{out}\n')

    # 下面这个Latex是正确的，不应该被修改
    latex = r'\left\{\begin{array}{ll}1 & \text { if } x>0 \\0 & \text { if } x \leq 0\end{array}\right.'
    out = post_post_process_latex(latex)
    assert out == latex
    print(f'{latex}\n==\n{out}\n')
