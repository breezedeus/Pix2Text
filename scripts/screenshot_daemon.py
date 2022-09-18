# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).

# 安装 pyperclip
# > pip install pyperclip

import os
import time
import glob

import pyperclip as pc

from pix2text import set_logger, Pix2Text, render_html
logger = set_logger(log_level='DEBUG')

SCREENSHOT_DIR = os.getenv(
    "SCREENSHOT_DIR", '/Users/king/Pictures/screenshot_from_xnip'
)

thresholds = {
    'formula2general': 0.65,  # 如果识别为 `formula` 类型，但得分小于此阈值，则改为 `general` 类型
    'english2general': 0.75,  # 如果识别为 `english` 类型，但得分小于此阈值，则改为 `general` 类型
}
P2T = Pix2Text(thresholds=thresholds)


def get_newest_fp_time(screenshot_dir):
    fn_list = glob.glob1(screenshot_dir, '*g')
    fp_list = [os.path.join(screenshot_dir, fn) for fn in fn_list]
    if not fp_list:
        return None, None
    fp_list.sort(key=lambda fp: os.path.getmtime(fp), reverse=True)
    return fp_list[0], os.path.getmtime(fp_list[0])


def recognize(screenshot_dir, delta_interval):
    while True:
        newest_fp, newest_mod_time = get_newest_fp_time(screenshot_dir)
        if (
            newest_mod_time is not None
            and time.time() - newest_mod_time < delta_interval
        ):
            logger.info(f'analyzing screenshot file {newest_fp}')
            image_type, result = _recognize_newest(newest_fp)
            logger.info('image type: %s, image text: %s', image_type, result)
            if result:
                pc.copy(result)
            render_html(newest_fp, image_type, result, out_html_fp='out-text.html')
        time.sleep(1)


def _recognize_newest(newest_fp):
    res = P2T.recognize(newest_fp)
    return res['image_type'], res['text']


if __name__ == '__main__':
    recognize(SCREENSHOT_DIR, 1.05)
