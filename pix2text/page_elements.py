# coding: utf-8
import dataclasses
from pathlib import Path
from typing import Sequence, Any, Union, Optional

from PIL import Image

from .table_ocr import visualize_cells
from .utils import merge_line_texts
from .layout_parser import ElementType


@dataclasses.dataclass
class Element(object):
    id: str
    box: Sequence
    text: Optional[str]
    meta: Any
    type: ElementType
    total_img: Image.Image
    score: float = 1.0
    kwargs: dict = dataclasses.field(default_factory=dict)

    def __init__(
        self,
        id: str,
        box: Sequence,
        meta: Any,
        type: ElementType,
        total_img: Image.Image,
        score: float,
        text: Optional[str] = None,
        configs: Optional[dict] = None,
    ):
        self.total_img = total_img
        self.id = id
        self.box = box
        self.meta = meta
        self.type = type
        self.score = score
        self.kwargs = configs or {}

        if self.meta is not None and text is None:
            self.text = self._meta_to_text()
        else:
            self.text = text

    def to_dict(self):
        return dataclasses.asdict(self)

    def _meta_to_text(self) -> str:
        if self.type in (ElementType.TEXT, ElementType.TITLE):
            embed_sep = self.kwargs.get('embed_sep', (' $', '$ '))
            isolated_sep = self.kwargs.get('isolated_sep', ('$$\n', '\n$$'))
            line_sep = self.kwargs.get('line_sep', '\n')
            auto_line_break = self.kwargs.get('auto_line_break', True)
            if self.type == ElementType.TITLE:
                for box_info in self.meta:
                    if box_info['type'] == 'isolated':
                        box_info['type'] = 'embedding'
            outs = merge_line_texts(
                self.meta, auto_line_break, line_sep, embed_sep, isolated_sep
            )
        elif self.type == ElementType.FORMULA:
            if isinstance(self.meta, dict):
                outs = self.meta['text']
            elif isinstance(self.meta, list):
                outs = [one['text'] for one in self.meta]
        elif self.type == ElementType.TABLE:
            outs = '\n'.join(self.meta.get('markdown', []))
        else:
            outs = ''

        return outs

    def __repr__(self) -> str:
        return f"Element(box={self.box}, text={self.text}, meta={self.meta}, type={self.type.name}, score={self.score})"

    def __str__(self) -> str:
        return self.__repr__()

    def __lt__(self, other) -> bool:
        """
        Adapted from https://github.com/SVJLucas/Scanipy/blob/main/scanipy/elements/element.py.
        Less than operator for Element objects.

        Args:
            other (Element): Another Element object.

        Returns:
            bool: True if this Element is "less than" the other, False otherwise.

        Raises:
            TypeError: If 'other' is not an instance or subclass of Element.
        """

        if not isinstance(other, Element):
            raise TypeError("other must be an instance or subclass of ElementOutput")

        return self._column_before(other) or (
            self._same_column(other) and (self.box[1] < other.box[1])
        )

    def _column_before(self, other) -> bool:
        """
        Adapted from https://github.com/SVJLucas/Scanipy/blob/main/scanipy/elements/element.py.
        Check if this Element is in a column before the other Element.

        Args:
            other (Element): Another Element object.

        Returns:
            bool: True if in a column before, False otherwise.
        """

        if not isinstance(other, Element):
            raise TypeError("other must be an instance or subclass of ElementOutput")

        max_width = max(box_width(self.box), box_width(other.box))
        return self.box[0] < other.box[0] - max_width / 2

    def _same_column(self, other) -> bool:
        """
        Check if this Element is in the same column as the other Element.

        Args:
            other (Element): Another Element object.

        Returns:
            bool: True if in the same column, False otherwise.
        """

        if not isinstance(other, Element):
            raise TypeError("other must be an instance or subclass of ElementOutput")

        max_width = max(box_width(self.box), box_width(other.box))
        return abs(self.box[0] - other.box[0]) < max_width / 2


def box_width(box):
    return box[2] - box[0]


class Page(object):
    id: str
    elements: Sequence[Element]
    config: dict

    def __init__(self, id: str, elements: Sequence[Element], config=None):
        self.id = id
        self.elements = elements
        self.config = config or {}

    def __repr__(self) -> str:
        return f"Page(id={self.id}, elements={self.elements})"

    def to_markdown(self, out_dir: Union[str, Path]):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        self.elements.sort()
        text_outs = []
        for element in self.elements:
            text_outs.append(self._ele_to_markdown(element, out_dir))
        md_out = '\n\n'.join(text_outs)
        with open(out_dir / f'output.md', 'w') as f:
            f.write(md_out)

    def _ele_to_markdown(self, element: Element, out_dir: Union[str, Path]):
        type = element.type
        text = element.text
        if type in (ElementType.TEXT, ElementType.TABLE):
            if type == ElementType.TABLE:
                visualize_cells(
                    element.total_img.crop(element.box),
                    element.meta['cells'][0],
                    out_dir / f'{element.id}.png',
                )
            return text
        elif type == ElementType.TITLE:
            return f'## {text}'
        elif type == ElementType.FORMULA:
            isolated_sep = self.config.get('isolated_sep', ('$$\n', '\n$$'))
            return isolated_sep[0] + text + isolated_sep[1]
        elif type == ElementType.FIGURE:
            out_figure_dir = out_dir / 'figures'
            out_figure_dir.mkdir(exist_ok=True, parents=True)
            out_path = out_figure_dir / f'{element.id}-{type.name}.jpg'
            element.total_img.crop(element.box).save(str(out_path))

            _url = self._map_path_to_url(out_path, out_dir)
            return f'![{self.id}]({_url})'
        return ''

    def _map_path_to_url(self, path: Path, out_dir: Path):
        return path.relative_to(out_dir)
