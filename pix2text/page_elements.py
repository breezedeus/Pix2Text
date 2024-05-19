# coding: utf-8
import dataclasses
from pathlib import Path
import re
from typing import Sequence, Any, Union, Optional

from PIL import Image

from .table_ocr import visualize_cells
from .utils import merge_line_texts, smart_join
from .layout_parser import ElementType


@dataclasses.dataclass
class Element(object):
    id: str
    box: Sequence
    text: Optional[str]
    meta: Any
    type: ElementType
    total_img: Image.Image
    isolated: bool = False
    col_number: int = -1
    score: float = 1.0
    spellchecker = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def __init__(
        self,
        *,
        id: str,
        box: Sequence,
        isolated: bool,
        col_number: int,
        meta: Any,
        type: ElementType,
        total_img: Image.Image,
        score: float,
        text: Optional[str] = None,
        spellchecker=None,
        configs: Optional[dict] = None,
    ):
        self.total_img = total_img
        self.id = id
        self.box = box
        self.isolated = isolated
        self.col_number = col_number
        self.meta = meta
        self.type = type
        self.score = score
        self.spellchecker = spellchecker
        self.kwargs = configs or {}

        if self.meta is not None and text is None:
            self.text = self._meta_to_text()
        else:
            self.text = text

        if self.isolated:
            self.text = self.text + '\n'

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
                    if box_info.get('type', 'text') == 'isolated':
                        box_info['type'] = 'embedding'
            outs = merge_line_texts(
                self.meta,
                auto_line_break,
                line_sep,
                embed_sep,
                isolated_sep,
                self.spellchecker,
            )
            if self.type == ElementType.TITLE:
                outs = smart_join(outs.split('\n'), self.spellchecker)
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
        return f"Element({self.to_dict()})"

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
            raise TypeError("other must be an instance or subclass of Element")

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
            raise TypeError("other must be an instance or subclass of Element")

        return self.col_number < other.col_number
        # max_width = max(box_width(self.box), box_width(other.box))
        # return self.box[0] < other.box[0] - max_width / 2

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

        return self.col_number == other.col_number
        # max_width = max(box_width(self.box), box_width(other.box))
        # return abs(self.box[0] - other.box[0]) < max_width / 2


def box_width(box):
    return box[2] - box[0]


class Page(object):
    number: int
    id: str
    elements: Sequence[Element]
    config: dict
    spellchecker = None

    def __init__(
        self,
        *,
        number: int,
        elements: Sequence[Element],
        id: Optional[str] = None,
        spellchecker=None,
        config=None,
    ):
        self.number = number
        self.id = id or str(number)
        self.elements = elements
        self.spellchecker = spellchecker
        self.config = config or {}

    def __repr__(self) -> str:
        return f"Page(id={self.id}, number={self.number}, elements={self.elements})"

    def to_markdown(
        self,
        out_dir: Union[str, Path],
        root_url: Optional[str] = None,
        markdown_fn: Optional[str] = 'output.md',
    ) -> str:
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        self.elements.sort()
        if not self.elements:
            return ''
        md_out = self._ele_to_markdown(self.elements[0], root_url, out_dir)
        for idx, element in enumerate(self.elements[1:]):
            prev_element = self.elements[idx]
            cur_txt = self._ele_to_markdown(element, root_url, out_dir)
            if (
                prev_element.col_number + 1 == element.col_number
                and prev_element.type == element.type
                and prev_element.type in (ElementType.TEXT, ElementType.TITLE)
                and md_out
                and md_out[-1] != '\n'
                and cur_txt
                and cur_txt[0] != '\n'
            ):
                # column continuation
                md_out = smart_join([md_out, cur_txt], self.spellchecker)
            else:
                md_out += '\n\n' + cur_txt

        line_sep = '\n'
        md_out = re.sub(
            rf'{line_sep}{{2,}}', f'{line_sep}{line_sep}', md_out
        )  # 把2个以上的连续 '\n' 替换为 '\n\n'
        if markdown_fn:
            with open(out_dir / markdown_fn, 'w', encoding='utf-8') as f:
                f.write(md_out)
        return md_out

    def _ele_to_markdown(
        self, element: Element, root_url: Optional[str], out_dir: Union[str, Path]
    ):
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
            return isolated_sep[0] + text.strip() + isolated_sep[1]
        elif type == ElementType.FIGURE:
            out_figure_dir = out_dir / 'figures'
            out_figure_dir.mkdir(exist_ok=True, parents=True)
            out_path = out_figure_dir / f'{element.id}-{type.name}.jpg'
            element.total_img.crop(element.box).save(str(out_path))

            _url = self._map_path_to_url(root_url, out_path, out_dir)
            return f'![]({_url})'
        return ''

    def _map_path_to_url(self, root_url: Optional[str], path: Path, out_dir: Path):
        rel_url = path.relative_to(out_dir)
        if root_url is not None:
            return f'{root_url}/{rel_url}'
        return str(rel_url)


class Document(object):
    number: int
    id: str
    pages: Sequence[Page]
    config: dict
    spellchecker = None

    def __init__(
        self,
        *,
        number: int,
        pages: Sequence[Page],
        id: Optional[str] = None,
        spellchecker=None,
        config=None,
    ):
        self.number = number
        self.id = id or str(number)
        self.pages = pages
        self.spellchecker = spellchecker
        self.config = config or {}

    def __repr__(self) -> str:
        return f"Document(id={self.id}, number={self.number}, pages={self.pages})"

    def to_markdown(
        self,
        out_dir: Union[str, Path],
        root_url: Optional[str] = None,
        markdown_fn: Optional[str] = 'output.md',
    ) -> str:
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        self.pages.sort(key=lambda page: page.number)
        if not self.pages:
            return ''
        md_out = self.pages[0].to_markdown(out_dir, root_url=root_url, markdown_fn=None)
        for idx, page in enumerate(self.pages[1:]):
            prev_page = self.pages[idx]
            cur_txt = page.to_markdown(out_dir, mroot_url=root_url, arkdown_fn=None)
            if (
                md_out
                and prev_page.elements
                and prev_page.elements[-1].type in (ElementType.TEXT, ElementType.TITLE)
                and page.elements
                and page.elements[0].type in (ElementType.TEXT, ElementType.TITLE)
                and md_out[-1] != '\n'
                and cur_txt
                and cur_txt[0] != '\n'
            ):
                # column continuation
                md_out = smart_join([md_out, cur_txt], self.spellchecker)
            else:
                md_out += '\n\n' + cur_txt

        line_sep = '\n'
        md_out = re.sub(
            rf'{line_sep}{{2,}}', f'{line_sep}{line_sep}', md_out
        )  # 把2个以上的连续 '\n' 替换为 '\n\n'
        if markdown_fn:
            with open(out_dir / markdown_fn, 'w', encoding='utf-8') as f:
                f.write(md_out)
        return md_out
