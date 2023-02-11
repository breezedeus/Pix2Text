# coding: utf-8

from pix2text import Pix2Text
from pix2text.latex_ocr import post_post_process_latex


def test_mfd():
    thresholds = {
        'formula2general': 0.65,  # 如果识别为 `formula` 类型，但得分小于此阈值，则改为 `general` 类型
        'english2general': 0.75,  # 如果识别为 `english` 类型，但得分小于此阈值，则改为 `general` 类型
    }
    config = dict(analyzer=dict(model_name='mfd'), thresholds=thresholds)
    p2t = Pix2Text.from_config(config)

    res = p2t.recognize(
        './docs/examples/zh1.jpg',
        use_analyzer=True,
        save_analysis_res='./analysis_res.jpg',
    )
    print(res)


def test_post_post_process():
    latex = r'\log(p(x\mid q(x)))+||\mathrm{sg}\left[z_{e}(x)\right]-e||_{2}^{2}+\beta\left||z_{e}(x)-\mathrm{sg}[e]|_{2}^{2}'
    out = post_post_process_latex(latex)
    # 其他的测试字符串 latex
