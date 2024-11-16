# coding: utf-8
import dataclasses
from copy import deepcopy
from pathlib import Path
import re
from typing import Sequence, Any, Union, Optional

from PIL import Image

from .table_ocr import visualize_cells
from .utils import merge_line_texts, smart_join, y_overlap, list2box
from .layout_parser import ElementType

FORMULA_TAG = '^[（\(]\d+(\.\d+)*[）\)]$'


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
        """
        Convert the Page to markdown.
        Args:
            out_dir (Union[str, Path]): The output directory.
            root_url (Optional[str]): The root url for the saved images in the markdown files.
            markdown_fn (Optional[str]): The markdown file name. Default is 'output.md'.

        Returns: The markdown string.

        """
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        self.elements = self._merge_isolated_formula_and_tag(self.elements)
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

    @classmethod
    def _merge_isolated_formula_and_tag(cls, elements):
        # 合并孤立的公式和公式标题
        # 对于每个公式标题，找到与它在同一行且在其左侧距离最近的孤立公式，并把它们合并
        isolated_formula = [
            item
            for item in elements
            if item.type == ElementType.FORMULA and item.isolated
        ]
        formula_caption = [
            item
            for item in elements
            if item.type == ElementType.TEXT and re.match(FORMULA_TAG, item.text)
        ]
        ele_ids = set([item.id for item in isolated_formula + formula_caption])
        remaining_elements = [item for item in elements if item.id not in ele_ids]
        for caption in formula_caption:
            caption_xmin, caption_ymin, caption_xmax, caption_ymax = caption.box
            min_dist = float('inf')
            nearest_formula = None
            for formula in isolated_formula:
                formula_xmin, formula_ymin, formula_xmax, formula_ymax = formula.box
                if (
                    caption.col_number == formula.col_number
                    and y_overlap(
                        list2box(*caption.box), list2box(*formula.box), key=None
                    )
                    >= 0.8
                ):
                    dist = caption_xmin - formula_xmax
                    if 0 <= dist < min_dist:
                        min_dist = dist
                        nearest_formula = formula
            if nearest_formula is not None:
                new_formula = deepcopy(nearest_formula)
                formula_xmin, formula_ymin, formula_xmax, formula_ymax = new_formula.box
                new_formula.box = [
                    min(caption_xmin, formula_xmin),
                    min(caption_ymin, formula_ymin),
                    max(caption_xmax, formula_xmax),
                    max(caption_ymax, formula_ymax),
                ]
                new_text = new_formula.text.strip() + ' \\tag{{{}}}'.format(
                    caption.text[1:-1]
                )
                new_formula.text = new_text
                if new_formula.meta and isinstance(new_formula.meta, dict):
                    new_formula.meta['text'] = new_text
                remaining_elements.append(new_formula)
                isolated_formula.remove(nearest_formula)
            else:  # not found
                remaining_elements.append(caption)
        return remaining_elements + isolated_formula


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
        """
        Convert the Document to markdown.
        Args:
            out_dir (Union[str, Path]): The output directory.
            root_url (Optional[str]): The root url for the saved images in the markdown files.
            markdown_fn (Optional[str]): The markdown file name. Default is 'output.md'.

        Returns: The markdown string.

        """
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        self.pages.sort(key=lambda page: page.number)
        if not self.pages:
            return ''
        md_out = self.pages[0].to_markdown(out_dir, root_url=root_url, markdown_fn=None)
        for idx, page in enumerate(self.pages[1:]):
            prev_page = self.pages[idx]
            cur_txt = page.to_markdown(out_dir, root_url=root_url, markdown_fn=None)
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
