# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2026, [Breezedeus](https://www.breezedeus.com).

import logging
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Union

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

OFFICIAL_HF_ENDPOINT = 'https://huggingface.co'
DEFAULT_MIRROR_URLS = [OFFICIAL_HF_ENDPOINT, 'https://hf-mirror.com']


def dir_has_files(dir_path: Path) -> bool:
    if not dir_path.is_dir():
        return False
    for path in dir_path.rglob('*'):
        if path.name.startswith('.'):
            continue
        return True
    return False


class HuggingFaceDownloader:
    def __init__(
        self,
        mirror_urls: Optional[Union[str, Iterable[Optional[str]]]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.mirror_urls = self._normalize_mirror_urls(mirror_urls)
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _normalize_mirror_urls(
        mirror_urls: Optional[Union[str, Iterable[Optional[str]]]]
    ) -> List[str]:
        if mirror_urls is None:
            urls = list(DEFAULT_MIRROR_URLS)
        elif isinstance(mirror_urls, str):
            urls = [mirror_urls] + DEFAULT_MIRROR_URLS
        else:
            urls = list(mirror_urls)

        return urls or DEFAULT_MIRROR_URLS

    def download(
        self,
        repo_id: str,
        local_dir: Union[str, Path],
        repo_type: str = 'model',
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
    ) -> bool:
        if snapshot_download is None:
            self.logger.error('huggingface_hub not installed. Please install huggingface_hub.')
            return False

        local_dir = Path(local_dir)
        for endpoint in self.mirror_urls:
            if local_dir.exists():
                shutil.rmtree(str(local_dir))
            local_dir.mkdir(parents=True, exist_ok=True)

            kwargs = {
                'repo_id': repo_id,
                'repo_type': repo_type,
                'local_dir': str(local_dir),
                'local_dir_use_symlinks': False,
            }
            if endpoint:
                kwargs['endpoint'] = endpoint
            if allow_patterns is not None:
                kwargs['allow_patterns'] = allow_patterns
            if ignore_patterns is not None:
                kwargs['ignore_patterns'] = ignore_patterns

            self.logger.info('Attempting to download from %s with repo_id: %s', endpoint or OFFICIAL_HF_ENDPOINT, repo_id)
            try:
                snapshot_download(**kwargs)
            except Exception as exc:
                label = endpoint or OFFICIAL_HF_ENDPOINT
                self.logger.warning('Snapshot download failed via %s: %s', label, exc)
                continue

            if dir_has_files(local_dir):
                return True

            label = endpoint or OFFICIAL_HF_ENDPOINT
            self.logger.warning('No files downloaded via %s to %s', label, local_dir)

        return False
