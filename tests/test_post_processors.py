# coding: utf-8
from pix2text.latex_ocr import *


def test_remove_redundant_script():
    latex_strs = [
        ('^ { abc }', 'abc'),
        ('^ { { a + b } }', '{ a + b }'),
        ('_ { abc }', 'abc'),
        ('_ { { a + b } }', '{ a + b }'),
        ('\\sum _ { t = 1 } ^ { T }', '\\sum _ { t = 1 } ^ { T }'),
    ]

    for ori, res in latex_strs:
        assert remove_redundant_script(ori) == res


def test_remove_empty_text():
    latex_strs = [
        (
            'J _ { \\stackrel { \\arraycolsep } { 0 p t } { G } } ^ { }',
            'J _ { \\stackrel { \\arraycolsep } { 0 p t } { G } }',
        ),
        ('\\hat { }', ''),
        ('\\hat { } _ { } : h = 0. 5', ': h = 0. 5'),
        ('\\sum _ { t = 1 } ^ { T }', '\\sum _ { t = 1 } ^ { T }'),
    ]

    for ori, res in latex_strs:
        assert remove_empty_text(ori) == res


def test_remove_trailing_whitespace():
    latex_strs = [
        ('abc \\qquad \\qquad \\qquad', 'abc'),
        ('abc \\qquad \\quad \\qquad', 'abc'),
        ('abc \\qquad \\ \\quad \\qquad', 'abc'),
        ('abc \\, \\, \\, \\, \\, \\, \\,', 'abc'),
        ('f ^ { \\prime } \\ = \\ \\ ', 'f ^ { \\prime } \\ ='),
        ('\\sum _ { t = 1 } ^ { T }', '\\sum _ { t = 1 } ^ { T }'),
    ]

    for ori, res in latex_strs:
        assert remove_trailing_whitespace(ori) == res


def test_remove_unnecessary_spaces():
    latex_strs = [
        ('{ \\cal L }', '{\\cal L}'),  # 保留命令后紧跟大写字母的空格
        ('\\textbf {bold text}', '\\textbf{bold text}'),  # 移除命令后的空格
        ('a + b = c', 'a+b=c'),  # 数学模式内的空格被移除
        ('\\frac{ 1 }{ 2 }', '\\frac{1}{2}'),  # 移除大括号内的空格
        ('\\sum_{ i = 1 }^{ N }', '\\sum_{i=1}^{N}'),  # 移除下标和上标中的空格
        ('\\alpha \\, \\beta', '\\alpha\\, \\beta'),  # 保留显式间距调整命令的空格
        ('\\sqrt { x } + \\sqrt { y }', '\\sqrt{x}+\\sqrt{y}'),  # 移除大括号内的空格，保留操作符周围的空格
        ('\\textit {italic text} with space', '\\textit{italic text} with space'),  # 移除命令后的空格，保留文本中的空格
        ('\\mathrm { a b c }', '\\mathrm{a b c}'),  # 移除命令后的空格
        ('\\sum _ {t=1} ^ {T} 4 _ { 2 }', '\\sum_{t=1}^{T} 4_{2}'),
        ('\\sim q ( z | x )', '\\sim q ( z | x )'),
    ]

    for ori, res in latex_strs:
        assert remove_unnecessary_spaces(ori) == res


def test_fix_latex():
    latex_strs = [
        ('\\left \\frac{1}{2}', '\\frac{1}{2}'),
        ('\\left ( \\frac{1}{2} + \\left \\frac{1}{2} \\right )', '\\left ( \\frac{1}{2} + \\frac{1}{2} \\right )'),
        ('\\left( \\frac{1}{2} + \\left \\frac{1}{2} \\right)', '\\left( \\frac{1}{2} + \\frac{1}{2} \\right)'),
    ]

    for ori, res in latex_strs:
        assert fix_latex(ori) == res


def test_replace_illegal_symbols():
    latex_strs = [
        (r'a^2 \.', r'a^2 \ .'),
        (r'a^2 \= \.', r'a^2 \ = \ .'),
        (r'a^2 \- \.', r'a^2 \ - \ .'),
        (r'a^2 \~ \.', r'a^2 \ ~ \ .'),
    ]

    for ori, res in latex_strs:
        assert replace_illegal_symbols(ori) == res