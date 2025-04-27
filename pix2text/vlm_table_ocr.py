# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).

import os
from typing import Union, Optional, Dict, Any, List
from copy import deepcopy
from pathlib import Path

from PIL import Image
import numpy as np

from .utils import read_img



# Default VLM prompt for table recognition
TABLE_PROMPT = """
首先识别图片中的文字是什么语言，然后再把图片中的表格转换成Markdown格式表示， 数学公式使用tex表示。
注意：
- 不要出现任何多余的文字
- 行内内嵌公式使用$符号包裹
- 独立行公式使用$$符号包裹
- 表格中的每行开头和结尾都要有|
输出格式示例：
## text_language
en
## text_content
```
|---|---|
| 1 | line1 |
| 2 | square: $a^2$ |
| 3 | $$r^2$$ |
```
)
"""


class VlmTableOCR(object):
    """
    Implements table extraction using Vision Language Models.
    This class uses the same interface as TableOCR but leverages VLM capabilities.
    """

    def __init__(
        self,
        vlm=None,
        **kwargs,
    ):
        """
        Initialize a VlmTableOCR object.

        Args:
            vlm: Vision Language Model instance for table recognition
            **kwargs: Additional parameters
        """
        if vlm is None:
            raise ValueError("vlm must be provided")

        self.vlm = vlm

    @classmethod
    def from_config(
        cls,
        configs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Create a VlmTableOCR instance from configuration.

        Args:
            vlm: Vision Language Model instance
            configs (Optional[dict], optional): Configuration dictionary
            **kwargs: Additional parameters

        Returns:
            VlmTableOCR: An instance of VlmTableOCR
        """
        from .vlm_api import Vlm

        # Combine configs with any additional kwargs
        all_kwargs = kwargs.copy()
        if configs:
            all_kwargs.update(configs)
        
        vlm = Vlm(
            model_name=all_kwargs.pop("model_name", None),
            api_key=all_kwargs.pop("api_key", None),
        )

        return cls(
            vlm=vlm,
            **all_kwargs
        )

    def recognize(
        self,
        img: Union[str, Path, Image.Image],
        *,
        prompt: Optional[str] = TABLE_PROMPT,
        out_objects=False,
        out_cells=False,
        out_html=False,
        out_csv=False,
        out_markdown=True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Recognize tables from an image using VLM.

        Args:
            img: Input image (path, PIL.Image)
            prompt (Optional[str]): Custom prompt for VLM
            out_objects (bool): Whether to output objects
            out_cells (bool): Whether to output cells
            out_html (bool): Whether to output HTML
            out_csv (bool): Whether to output CSV
            out_markdown (bool): Whether to output Markdown
            **kwargs: Additional parameters
                * resized_shape (int): Resize shape for large images
                * save_analysis_res (str): Save the parsed result image in this file

        Returns:
            Dict[str, Any]: Dictionary containing recognized table data in requested formats
        """
        out_formats = {}

        if not (out_objects or out_cells or out_html or out_csv or out_markdown):
            print("No output format specified")
            return out_formats

        if not isinstance(img, (str, Path, Image.Image)):
            raise ValueError("img must be a path or PIL.Image")

        # Process with VLM
        try:
            vlm_result = self.vlm(
                img_path=img,
                prompt=prompt,
                auto_resize=True,
                resized_shape=kwargs.get("resized_shape", 768),
                **kwargs,
            )

            markdown_text = vlm_result.get("text", "")

            # For markdown output
            if out_markdown:
                out_formats["markdown"] = [markdown_text]

            # For HTML output (convert from markdown if needed)
            if out_html:
                try:
                    import markdown

                    html_text = markdown.markdown(markdown_text, extensions=["tables"])
                    # Extract just the table HTML
                    if "<table>" in html_text:
                        table_html = html_text[
                            html_text.find("<table>") : html_text.rfind("</table>") + 8
                        ]
                        out_formats["html"] = [table_html]
                    else:
                        out_formats["html"] = [
                            "<table><tr><td>Failed to convert to HTML</td></tr></table>"
                        ]
                except ImportError:
                    out_formats["html"] = [
                        "<table><tr><td>Markdown conversion library not available</td></tr></table>"
                    ]

            # For CSV output (convert from markdown if needed)
            if out_csv:
                try:
                    import pandas as pd
                    import io

                    # Simple markdown table to CSV conversion
                    lines = [
                        line.strip()
                        for line in markdown_text.split("\n")
                        if line.strip()
                    ]
                    cleaned_lines = []

                    for line in lines:
                        if line.startswith("|") and line.endswith("|"):
                            # Remove the first and last | and split by |
                            cells = [cell.strip() for cell in line[1:-1].split("|")]
                            cleaned_lines.append(",".join(cells))

                    if cleaned_lines and "---" in cleaned_lines[1]:
                        # Remove the separator line (---|---|---)
                        cleaned_lines.pop(1)

                    csv_content = "\n".join(cleaned_lines)
                    out_formats["csv"] = [csv_content]
                except Exception as e:
                    out_formats["csv"] = [f"Error converting to CSV: {str(e)}"]

            # For cellular representation (simplified for VLM)
            if out_cells:
                raise NotImplementedError(
                    "Cellular representation is not implemented for VLMTableOCR."
                )

            # For objects (simplified for VLM)
            if out_objects:
                raise NotImplementedError(
                    "Object representation is not implemented for VLMTableOCR."
                )

        except Exception as e:
            print(f"Error recognizing table: {e}")
            if out_markdown:
                out_formats["markdown"] = ["Error processing table with VLM"]

        return out_formats
