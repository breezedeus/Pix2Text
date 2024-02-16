#!/usr/bin/env python3
# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).

import os
from setuptools import find_packages, setup
from pathlib import Path

PACKAGE_NAME = "pix2text"

here = Path(__file__).parent

long_description = (here / "README.md").read_text(encoding="utf-8")

about = {}
exec(
    (here / PACKAGE_NAME.replace('.', os.path.sep) / "__version__.py").read_text(
        encoding="utf-8"
    ),
    about,
)

required = [
    "click",
    "tqdm",
    "numpy",
    "opencv-python",
    "cnocr[ort-cpu]>=2.3.0.1",
    "cnstd>=1.2.3.5",
    "pillow",
    "torch",
    "torchvision",
    "transformers>=4.37.0",
    "optimum[onnxruntime]",
]
extras_require = {
    "multilingual": ["easyocr"],
    "dev": ["pip-tools", "pytest"],
    "serve": ["uvicorn[standard]", "fastapi", "python-multipart", "pydantic"],
}

entry_points = """
[console_scripts]
p2t = pix2text.cli:cli
"""

setup(
    name=PACKAGE_NAME,
    version=about['__version__'],
    description="An Open-Source Python3 tool for Optical Character Recognition (OCR) "
    "and LaTeX expression extraction from images; a Free Alternative to Mathpix",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='breezedeus',
    author_email='breezedeus@163.com',
    license='MIT',
    url='https://github.com/breezedeus/pix2text',
    platforms=["Mac", "Linux", "Windows"],
    packages=find_packages(),
    include_package_data=True,
    # data_files=[('', ['pix2text/latex_config.yaml',],)],
    entry_points=entry_points,
    install_requires=required,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
