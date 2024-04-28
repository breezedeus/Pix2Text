# coding: utf-8

import requests


def main():
    url = 'http://0.0.0.0:8503/pix2text'

    image_fp = 'docs/examples/page2.png'
    # image_fp = 'docs/examples/mixed.jpg'
    # image_fp = 'docs/examples/math-formula-42.png'
    # image_fp = 'docs/examples/english.jpg'
    data = {
        "file_type": "page",
        "resized_shape": 768,
        "embed_sep": " $,$ ",
        "isolated_sep": "$$\n, \n$$"
    }
    files = {
        "image": (image_fp, open(image_fp, 'rb'), 'image/jpeg')
    }

    r = requests.post(url, data=data, files=files)

    outs = r.json()['results']
    out_md_dir = r.json()['output_dir']
    if isinstance(outs, str):
        only_text = outs
    else:
        only_text = '\n'.join([out['text'] for out in outs])
    print(f'{only_text=}')
    print(f'{out_md_dir=}')


if __name__ == '__main__':
    main()
