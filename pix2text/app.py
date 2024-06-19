# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).

from PIL import Image
import streamlit as st

from pix2text import set_logger, Pix2Text

logger = set_logger()
st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True)
def get_model():
    return Pix2Text()


def main():
    p2t = get_model()

    title = '开源工具 <a href="https://github.com/breezedeus/pix2text">Pix2Text</a> Demo'
    st.markdown(f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)

    subtitle = '作者：<a href="https://github.com/breezedeus">breezedeus</a>； ' \
               '欢迎加入 <a href="https://cnocr.readthedocs.io/zh-cn/stable/contact/">交流群</a>'

    st.markdown(f"<div style='text-align: center;'>{subtitle}</div>", unsafe_allow_html=True)
    st.markdown('')
    st.subheader('选择待识别图片')
    content_file = st.file_uploader('', type=["png", "jpg", "jpeg", "webp"])
    if content_file is None:
        st.stop()

    try:
        img = Image.open(content_file).convert('RGB')
        img.save('ori.jpg')

        out = p2t(img)
        logger.info(out)
        st.markdown('##### 原始图片：')
        cols = st.columns([1, 3, 1])
        with cols[1]:
            st.image(content_file)

        st.subheader('识别结果：')
        st.markdown(f"* **图片类型**：{out['image_type']}")
        st.markdown("* **识别内容**：")

        cols = st.columns([1, 3, 1])
        with cols[1]:
            st.text(out['text'])

            if out['image_type'] == 'formula':
                st.markdown(f"$${out['text']}$$")

    except Exception as e:
        st.error(e)


if __name__ == '__main__':
    main()
