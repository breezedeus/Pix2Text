# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, Form, HTTPException

from pix2text import set_logger, read_img, Pix2Text

logger = set_logger(log_level='DEBUG')

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Pix2Text Server!"}


class Pix2TextResponse(BaseModel):
    status_code: int = 200
    results: Union[str, List[Dict[str, Any]]]
    output_dir: Optional[str] = None

    def dict(self, **kwargs):
        the_dict = deepcopy(super().dict())
        return the_dict


@app.post("/pix2text")
async def ocr(
    image: UploadFile,
    file_type: str = Form(default='text_formula'),
    resized_shape: str = Form(default=768),
    embed_sep: str = Form(default=' $,$ '),
    isolated_sep: str = Form(default='$$\n, \n$$'),
) -> Dict[str, Any]:
    # curl 调用方式：
    # $ curl -F image=@docs/examples/english.jpg --form 'image_type=mixed' --form 'resized_shape=768' \
    #       http://0.0.0.0:8503/pix2text
    global P2T, OUTPUT_MD_ROOT_DIR
    if file_type not in ('text', 'formula', 'text_formula', 'page'):
        raise HTTPException(status_code=400, detail='file_type must be one of "text", "formula", "text_formula", "page"')

    img_file = image.file
    fn = Path(image.filename)
    img0 = read_img(img_file, return_type='Image')
    embed_sep = embed_sep.split(',')
    isolated_sep = isolated_sep.split(',')
    # use_analyzer = use_analyzer.lower() != 'false' if isinstance(use_analyzer, str) else use_analyzer

    params = dict(resized_shape=int(resized_shape), return_text=True)
    if len(embed_sep) == 2:
        params['embed_sep'] = embed_sep
    if len(isolated_sep) == 2:
        params['isolated_sep'] = isolated_sep

    logger.info(f'input {params=}')

    res = P2T.recognize(img0, file_type=file_type, **params)
    output_dir = None
    if file_type in ('pdf', 'page'):
        output_dir = str(OUTPUT_MD_ROOT_DIR / f'{fn.stem}-{time.time()}')
        res = res.to_markdown(output_dir)
    logger.info(f'output {res=}')

    return Pix2TextResponse(results=res, output_dir=output_dir).dict()


def start_server(
    p2t_config, output_md_root_dir, host='0.0.0.0', port=8503, reload=False, **kwargs
):
    global P2T, OUTPUT_MD_ROOT_DIR
    OUTPUT_MD_ROOT_DIR = Path(output_md_root_dir)
    OUTPUT_MD_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    P2T = Pix2Text.from_config(**p2t_config)
    import uvicorn

    uvicorn.run(app, host=host, port=port, reload=reload, **kwargs)
