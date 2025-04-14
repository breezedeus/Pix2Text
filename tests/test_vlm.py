# coding=utf-8
import os
import dotenv

from pix2text import set_logger
from pix2text.vlm_api import Vlm
from pix2text.vlm_table_ocr import VlmTableOCR
from pix2text.text_formula_ocr import VlmTextFormulaOCR

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


def test_vlm_text_formula_ocr():
    img_path = "docs/examples/mixed.jpg" 
    img_path = "docs/examples/vietnamese.jpg" 
    
    vlm_text_formula_ocr = VlmTextFormulaOCR.from_config(
        model_name=os.getenv("GEMINI_MODEL"),
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = vlm_text_formula_ocr.recognize(img_path, return_text=False)
    
    # Print the result
    print("识别结果:")
    print(result)
    
    # 可以进一步测试提取公式部分
    if hasattr(vlm_text_formula_ocr, 'extract_formula'):
        formula = vlm_text_formula_ocr.extract_formula(result)
        print("提取的公式:")
        print(formula)