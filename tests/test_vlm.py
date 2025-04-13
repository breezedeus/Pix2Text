# coding=utf-8
import os
import dotenv

from pix2text import set_logger
from pix2text.vlm_api import Vlm
from pix2text.vlm_table_ocr import VlmTableOCR

logger = set_logger()
# Load environment variables from .env file
dotenv.load_dotenv()


def init_vlm():
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL")

    return Vlm(
        model_name=GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
    )


def test_vlm_api():
    img_path = "/Users/king/Documents/WhatIHaveDone/Test/pix2text/docs/feedbacks/2024-09-09.jpg"
    img_path = "/Users/king/Documents/WhatIHaveDone/Test/pix2text/docs/examples/table-cn.jpg"  # 表格，中文
    img_path = "/Users/king/Documents/WhatIHaveDone/Test/pix2text/docs/examples/ch_tra1.jpg"  # 繁体中文
    # img_path = "/Users/king/Documents/WhatIHaveDone/Test/pix2text/docs/examples/hw-formula5.jpg"  # 手写公式
    img_path = "docs/examples/hw-zh-en.jpg"  # 手写文字
    img_path = "docs/examples/hw-zh1.jpg"  # 手写文字
    img_path = "docs/examples/hw-zh2.jpg"  # 手写文字
    img_path = "docs/examples/hw-zh3.jpg"  # 手写文字
    img_path = ["docs/examples/hw-zh1.jpg", "docs/examples/hw-zh3.jpg"]  # 手写文字

    vlm = init_vlm()
    result = vlm(img_path, auto_resize=True)

    # Print the result
    print(result)


def test_vlm_table_ocr():
    img_path = "/Users/king/Documents/WhatIHaveDone/Test/pix2text/docs/examples/table-cn.jpg"  # 表格，中文

    vlm_table_ocr = VlmTableOCR.from_config(
        model_name=os.getenv("GEMINI_MODEL"),
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = vlm_table_ocr.recognize(img_path)

    # Print the result
    print(result)