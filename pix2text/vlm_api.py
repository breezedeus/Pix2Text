# coding: utf-8
import io
import requests
import base64
import logging
from typing import Optional, Union

from PIL import Image
from litellm import completion

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


PROMPT = """
首先识别图片中的文字是什么语言，然后再把图片中的文字或数学公式转换成Markdown格式表示， 数学公式使用tex表示。
注意：
- 不要出现任何多余的文字
- 行内内嵌公式使用$符号包裹
- 独立行公式使用$$符号包裹
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
        img_path: Union[str, Image.Image],
        *,
        prompt: str = PROMPT,
        auto_resize: bool = True,
        **kwargs,
    ):
        """Call the VLM API to convert image to text.
        Args:
            img_path (str): Path to the image file.
            prompt (str): Prompt for the API.
            auto_resize (bool): Whether to automatically resize large images.
            **kwargs: Additional arguments for the API call.
        Returns:
            str: The text extracted from the image. None if an error occurs.
        """
        if not isinstance(img_path, (str, Image.Image)):
            raise ValueError("img_path must be a string or PIL Image object")
        base64_image = encode_image(img_path, auto_resize=auto_resize)
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
        try:
            response = completion(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": content},
                ],
                api_key=self.api_key,
                **kwargs,
            )

            # Extract the response content
            out = response.get("choices", [{}])[0].get("message", {}).get("content")
        except Exception as exc:
            logger.error("An error occurred: %s", exc)
            out = None

        return out

    def __repr__(self):
        return f"Vlm(model_name={self.model_name})"

    def __str__(self):
        return f"Vlm(model_name={self.model_name})"
