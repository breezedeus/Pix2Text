# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).

from copy import deepcopy
from typing import Dict, Any

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile
from PIL import Image

from pix2text import set_logger, Pix2Text

logger = set_logger(log_level='DEBUG')

app = FastAPI()
P2T = Pix2Text()


class OcrResponse(BaseModel):
    status_code: int = 200
    results: Dict[str, Any]

    def dict(self, **kwargs):
        the_dict = deepcopy(super().dict())
        return the_dict


@app.get("/")
async def root():
    return {"message": "Welcome to Pix2Text Server!"}


@app.post("/pix2text")
async def ocr(image: UploadFile) -> Dict[str, Any]:
    image = image.file
    image = Image.open(image).convert('RGB')
    res = P2T(image)

    return OcrResponse(results=res).dict()
