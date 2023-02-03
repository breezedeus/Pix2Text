# coding: utf-8

import requests


def main():
    url = 'http://0.0.0.0:8503/pix2text'

    image_fp = 'docs/examples/mixed.jpg'
    data = {
        "use_analyzer": True,
        "resized_shape": 600,
        "embed_sep": " $,$ ",
        "isolated_sep": "$$\n, \n$$"
    }
    files = {
        "image": (image_fp, open(image_fp, 'rb'))
    }

    r = requests.post(url, data=data, files=files)

    outs = r.json()['results']
    only_text = '\n'.join([out['text'] for out in outs])
    print(f'{only_text=}')


if __name__ == '__main__':
    main()
