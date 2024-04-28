# coding: utf-8
from pix2text.doc_xl_layout import DocXLayoutParser
from pix2text.utils import set_logger

logger = set_logger()


def test_doc_xl_layout_parser():
    # model_fp = os.path.expanduser('~/.pix2text/1.0/doc_xl_layout/DocXLayout_231012.pth')
    img_fp = '/Users/king/Documents/WhatIHaveDone/Test/pix2text/docs/examples/page2.png'
    layout_parser = DocXLayoutParser(debug=1)
    out = layout_parser.parse(img_fp)
    print(out)
