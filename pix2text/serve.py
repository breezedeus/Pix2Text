# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).

from copy import deepcopy
from typing import Dict, List, Any

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, Form

from pix2text import set_logger, read_img, Pix2Text

logger = set_logger(log_level='DEBUG')

app = FastAPI()
P2T = Pix2Text(analyzer_config=dict(model_name='mfd'))


class Pix2TextRequest(BaseModel):
    image: UploadFile
    resized_shape: int = 600
    use_analyzer: bool = True
    embed_sep: tuple = (' $$', '$$ ')
    isolated_sep: tuple = ('\n', '\n')


class Pix2TextResponse(BaseModel):
    status_code: int = 200
    results: List[Dict[str, Any]]

    def dict(self, **kwargs):
        the_dict = deepcopy(super().dict())
        return the_dict


@app.get("/")
async def root():
    return {"message": "Welcome to Pix2Text Server!"}


@app.post("/pix2text")
async def ocr(
    image: UploadFile,
    use_analyzer: str = Form(default=True),
    resized_shape: str = Form(default=608),
    embed_sep: str = Form(default=' $,$ '),
    isolated_sep: str = Form(default='$$\n, \n$$'),
) -> Dict[str, Any]:
    # curl 调用方式：
    # $ curl -F image=@docs/examples/english.jpg --form 'use_analyzer=false' --form 'resized_shape=700' \
    #       http://0.0.0.0:8503/pix2text
    image = image.file
    image = read_img(image, return_type='Image')
    embed_sep = embed_sep.split(',')
    isolated_sep = isolated_sep.split(',')
    use_analyzer = use_analyzer.lower() != 'false' if isinstance(use_analyzer, str) else use_analyzer

    params = dict(
        use_analyzer=use_analyzer, resized_shape=int(resized_shape),
    )
    if len(embed_sep) == 2:
        params['embed_sep'] = embed_sep
    if len(isolated_sep) == 2:
        params['isolated_sep'] = isolated_sep

    logger.info(f'input {params=}')

    res = P2T(image, **params)
    for out in res:
        out['position'] = out['position'].tolist()
    logger.info(f'output {res=}')

    return Pix2TextResponse(results=res).dict()
