"""
Disk-based result cache to avoid duplicate/redundant API calls and compute.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Dict
from indic_pipeline.config import settings
from indic_pipeline.utils.logger import logger


class DiskCache:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, image_path: Path, engine_name: str, params: Optional[Dict[str, Any]] = None) -> str:
        hasher = hashlib.sha256()
        # Hash file content
        with open(image_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        hasher.update(engine_name.encode("utf-8"))
        if params:
            hasher.update(json.dumps(params, sort_keys=True).encode("utf-8"))
        return hasher.hexdigest()

    def get(self, image_path: Path, engine_name: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        key = self._compute_key(image_path, engine_name, params)
        cache_file = self.cache_dir / f"{engine_name}_{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    logger.debug(f"Cache hit for {image_path.name} with engine {engine_name}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
        return None

    def set(self, image_path: Path, engine_name: str, data: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> None:
        key = self._compute_key(image_path, engine_name, params)
        cache_file = self.cache_dir / f"{engine_name}_{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Error writing to cache file {cache_file}: {e}")

    def clear(self) -> None:
        count = 0
        for f in self.cache_dir.glob("*.json"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
        logger.info(f"Cleared {count} cached response files from {self.cache_dir}")


disk_cache = DiskCache()
