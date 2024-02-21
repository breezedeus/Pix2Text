# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).

from copy import deepcopy
from typing import Dict, List, Any, Union

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, Form

from pix2text import set_logger, read_img, Pix2Text

logger = set_logger(log_level='DEBUG')

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Pix2Text Server!"}


class Pix2TextResponse(BaseModel):
    status_code: int = 200
    results: Union[str, List[Dict[str, Any]]]

    def dict(self, **kwargs):
        the_dict = deepcopy(super().dict())
        return the_dict


@app.post("/pix2text")
async def ocr(
    image: UploadFile,
    image_type: str = Form(default='mixed'),
    resized_shape: str = Form(default=608),
    embed_sep: str = Form(default=' $,$ '),
    isolated_sep: str = Form(default='$$\n, \n$$'),
) -> Dict[str, Any]:
    # curl 调用方式：
    # $ curl -F image=@docs/examples/english.jpg --form 'image_type=mixed' --form 'resized_shape=768' \
    #       http://0.0.0.0:8503/pix2text
    global P2T
    image = image.file
    image = read_img(image, return_type='Image')
    embed_sep = embed_sep.split(',')
    isolated_sep = isolated_sep.split(',')
    # use_analyzer = use_analyzer.lower() != 'false' if isinstance(use_analyzer, str) else use_analyzer

    params = dict(resized_shape=int(resized_shape),)
    if len(embed_sep) == 2:
        params['embed_sep'] = embed_sep
    if len(isolated_sep) == 2:
        params['isolated_sep'] = isolated_sep

    logger.info(f'input {params=}')

    func = P2T.recognize
    if image_type == 'formula':
        func = P2T.recognize_formula
    elif image_type == 'text':
        func = P2T.recognize_text
    res = func(image, **params)
    if image_type == 'mixed':
        for out in res:
            out['position'] = out['position'].tolist()
    logger.info(f'output {res=}')

    return Pix2TextResponse(results=res).dict()


def start_server(p2t_config, host='0.0.0.0', port=8503, reload=False, **kwargs):
    global P2T
    P2T = Pix2Text(**p2t_config)
    import uvicorn

    uvicorn.run(app, host=host, port=port, reload=reload, **kwargs)
