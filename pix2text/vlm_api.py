# coding: utf-8
import io
import requests
import base64
import logging
from typing import Optional, Union, List
from pathlib import Path

import numpy as np
from PIL import Image
from litellm import completion, batch_completion

logger = logging.getLogger(__name__)


# Function to encode the image
def encode_image(
    image_path: Union[str, Image.Image],
    *,
    max_image_size: int = 768,
    auto_resize: bool = True,
) -> Optional[str]:
    """Encodes an image file or URL to base64 string with optional resizing.

    This function can handle three types of inputs:
        1. PIL Image object
        2. URL string (starting with http:// or https://)
        3. Local file path

    The function optionally resizes images that exceed the specified maximum dimension
    while maintaining the aspect ratio.

    Args:
        image_path (Union[str, PIL.Image.Image]): Path to image file, URL, or PIL Image object
        max_image_size (int, optional): Maximum dimension for image resize. Defaults to 768.
        auto_resize (bool, optional): Whether to automatically resize large images. Defaults to True.

    Returns:
        str: Base64 encoded string of the image data.
             Returns None if URL fetching fails.

    Example:
        >>> encoded = encode_image("path/to/image.jpg")
        >>> encoded = encode_image("https://example.com/image.jpg")
        >>> encoded = encode_image(pil_image_object)
    """
    if isinstance(image_path, Image.Image):
        img = image_path.convert("RGB")
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        image_data = buffered.getvalue()
    elif image_path.startswith(("http://", "https://")):
        response = requests.get(image_path)
        try:
            response.raise_for_status()  # Raise an error for bad responses
        except requests.exceptions.HTTPError as e:
            print(f"Failed to fetch image: {e}")
            return None
        image_data = response.content
    else:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
    if not auto_resize:
        return base64.b64encode(image_data).decode("utf-8")

    # 对分辨率太高的图片， 把其短边压缩到 max_image_size
    image = Image.open(io.BytesIO(image_data))
    width, height = image.size
    if width > max_image_size and height > max_image_size:
        if width > height:
            new_height = max_image_size
            new_width = int(width * max_image_size / height)
        else:
            new_height = int(height * max_image_size / width)
            new_width = max_image_size
        image = image.resize((new_width, new_height))
    # image.save("out-resize.png")
    # Convert image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def parse_content(content) -> dict:
    """Parse the content from the API response.
    Example: 
        convert '## text_language\nzh\n## text_content\n```\n## 写作。Writing. (50 points)\n写一写自己的一个爱好。\n```' to a structured dictionary:
        {
            "language": "zh",
            "content": "## 写作。Writing. (50 points)\n写一写自己的一个爱好。\n```"
        }

    Args:
        content (str): The content string to parse.

    Returns:
        dict: The parsed content, with keys "language" and "text".
            - language (str): The language of the content.
            - text (str): The text content, which may include Markdown formatting.
    """
    if not isinstance(content, str):
        raise ValueError("Content must be a string")
    splits = content.split("## text_content")
    if len(splits) != 2:
        raise ValueError("Content format is incorrect")
    parsed_str = splits[1].strip()
    # 去掉 开头的 ```.*\n 和结尾的 ```
    if parsed_str.startswith("```"):
        parsed_str = parsed_str[parsed_str.index("\n") + 1 :]
    if parsed_str.endswith("```"):
        parsed_str = parsed_str[: parsed_str.rindex("```")].strip()
    lang_splits = splits[0].split("## text_language")
    if len(lang_splits) != 2:
        raise ValueError("Language format is incorrect")
    lang = lang_splits[1].strip()

    return {
        "language": lang,
        "text": parsed_str,
    } 

PROMPT = """
首先识别图片中的文字是什么语言，然后再把图片中的文字或数学公式转换成Markdown格式表示， 数学公式使用tex表示。
注意：
- 不要出现任何多余的文字
- 行内内嵌公式使用`$...$`包裹
- 独立行公式使用`$$\n...\n$$`包裹
- 表格中的每行开头和结尾都要有|
- 段落标题前面使用合适数量的 #
输出格式示例：
## text_language
zh
## text_content
```
这是文字。内嵌公式：$x^2$。独立行公式：
$$
x^2 + y^2 = z^2
$$
```
"""


class Vlm(object):
    """
    VLM API for image-to-text conversion.
    This class uses the Litellm library to interact with the VLM API.
    """

    def __init__(self, *, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        self.api_key = api_key

    def __call__(
        self,
        img_path: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]] = None,
        *,
        prompt: str = PROMPT,
        resized_shape: int = 768,
        auto_resize: bool = True,
        parsing_func: Optional[callable] = parse_content,
        **kwargs,
    ) -> Union[dict, List[dict]]:
        """Call the VLM API to convert image to text.
        Args:
            img_path (Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]]): Path to the image file or files.
            prompt (str): Prompt for the API.
            auto_resize (bool): Whether to automatically resize large images.
            **kwargs: Additional arguments for the API call.
        Returns:
            dict or List[dict]: A dictionary for single image and list of dicts for multiple images. Each dict contains the text extracted from the image and the score:
                - text: Extracted text from the image.
                - score: Probability score of the extracted text.
        """
        single_image = False if isinstance(img_path, (list, tuple)) else True
        img_paths = [img_path] if single_image else img_path
        messages = []
        for img_path in img_paths:
            if isinstance(img_path, Path):
                img_path = str(img_path)
            if not isinstance(img_path, (str, Image.Image)):
                raise ValueError("img_path must be a string or PIL Image object")
            base64_image = encode_image(img_path, max_image_size=resized_shape, auto_resize=auto_resize)
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
            messages.append(
                [
                    {"role": "user", "content": content},
                ]
            )

        try:
            responses = batch_completion(
                model=self.model_name,
                messages=messages,
                api_key=self.api_key,
                **kwargs,
            )

            results = []
            for response in responses:
                # Extract the response content
                out = response.get("choices", [{}])[0].get("message", {}).get("content")
                logprob = response.get("choices", [{}])[0].get("logprobs")
                # to probability
                prob = float(np.exp(logprob)) if logprob else 0.0
                if parsing_func is None:
                    one_res = {
                        "text": out,
                        "score": prob,
                    }
                else:
                    try:
                        one_res = parsing_func(out)
                        one_res["score"] = prob
                    except Exception as exc:
                        logger.error("An error occurred while parsing the content: %s", exc)
                        one_res = {
                            "text": out,
                            "score": prob,
                        }
                results.append(one_res)
        except Exception as exc:
            logger.error("An error occurred: %s", exc)
            results = [{
                "text": "",
                "score": 0.0,
            } for _ in img_paths]

        return results[0] if single_image else results

    def __repr__(self):
        return f"Vlm(model_name={self.model_name})"

    def __str__(self):
        return f"Vlm(model_name={self.model_name})"
