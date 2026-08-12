import hashlib
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class CacheManager:
    """File-based cache with TTL support."""

    def __init__(
        self,
        cache_dir: str = ".research_cache",
        ttl_hours: int = 24,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.ttl = timedelta(hours=ttl_hours)
        self.metadata_file = self.cache_dir / "cache_metadata.json"

        self._load_metadata()

    def _load_metadata(self):
        if not self.metadata_file.exists():
            self.metadata = {}
            return

        try:
            with open(self.metadata_file, "r") as file:
                self.metadata = json.load(file)
        except Exception as exc:
            logger.warning(f"Failed to load cache metadata: {exc}")
            self.metadata = {}

    def _save_metadata(self):
        try:
            with open(self.metadata_file, "w") as file:
                json.dump(self.metadata, file)
        except Exception as exc:
            logger.warning(f"Failed to save cache metadata: {exc}")

    def _get_cache_key(self, query: str) -> str:
        return hashlib.md5(
            query.lower().encode()
        ).hexdigest()

    def _is_expired(
        self,
        timestamp: str,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        try:
            cached_time = datetime.fromisoformat(timestamp)

            item_ttl = (
                timedelta(seconds=ttl_seconds)
                if ttl_seconds is not None
                else self.ttl
            )

            return datetime.now() - cached_time > item_ttl

        except Exception:
            return True

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        cache_key = self._get_cache_key(query)

        if cache_key not in self.metadata:
            return None

        metadata = self.metadata[cache_key]
        ttl_seconds = metadata.get("ttl_seconds")

        if self._is_expired(
            metadata["timestamp"],
            ttl_seconds,
        ):
            self.delete(query)
            logger.info(f"Cache expired: {query[:50]}...")
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            logger.warning(
                f"Cache file missing: {cache_key}"
            )
            return None

        try:
            with open(cache_file, "rb") as file:
                result = pickle.load(file)

            logger.info(f"Cache hit: {query[:50]}...")
            return result

        except Exception as exc:
            logger.error(
                f"Failed to retrieve cache: {exc}"
            )
            return None

    def set(
        self,
        query: str,
        result: Dict[str, Any],
        ttl: Optional[int] = None,
    ):
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as file:
                pickle.dump(result, file)

            self.metadata[cache_key] = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "size": cache_file.stat().st_size,
                "ttl_seconds": ttl,
            }

            self._save_metadata()

            size_kb = cache_file.stat().st_size / 1024
            logger.info(
                f"Cached: {query[:50]}... ({size_kb:.1f}KB)"
            )

        except Exception as exc:
            logger.error(
                f"Failed to cache result: {exc}"
            )

    def delete(self, query: str) -> bool:
        cache_key = self._get_cache_key(query)

        if cache_key not in self.metadata:
            return False

        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            if cache_file.exists():
                cache_file.unlink()

            del self.metadata[cache_key]
            self._save_metadata()

            logger.info(
                f"Deleted cache: {query[:50]}..."
            )
            return True

        except Exception as exc:
            logger.error(
                f"Failed to delete cache: {exc}"
            )
            return False

    def clear_expired(self):
        expired_keys = []

        for cache_key, metadata in self.metadata.items():
            if self._is_expired(
                metadata["timestamp"],
                metadata.get("ttl_seconds"),
            ):
                expired_keys.append(cache_key)

        for cache_key in expired_keys:
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            try:
                if cache_file.exists():
                    cache_file.unlink()

                del self.metadata[cache_key]

            except Exception as exc:
                logger.warning(
                    f"Failed to delete expired cache "
                    f"{cache_key}: {exc}"
                )

        if expired_keys:
            self._save_metadata()
            logger.info(
                f"Cleared {len(expired_keys)} expired cache entries"
            )

    def clear_all(self):
        try:
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()

            self.metadata = {}
            self._save_metadata()

            logger.info("Cleared all cache")

        except Exception as exc:
            logger.error(
                f"Failed to clear cache: {exc}"
            )

    def get_stats(self) -> Dict[str, Any]:
        total_size = sum(
            file.stat().st_size
            for file in self.cache_dir.glob("*.pkl")
            if file.is_file()
        )

        return {
            "cached_queries": len(self.metadata),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
            "ttl_hours": self.ttl.total_seconds() / 3600,
        }


def cached_research_search(
    query: str,
    search_func,
    cache_dir: str = ".research_cache",
    ttl_hours: int = 24,
):
    cache = CacheManager(cache_dir, ttl_hours)

    cached_result = cache.get(query)

    if cached_result is not None:
        return cached_result

    result = search_func()
    cache.set(query, result)

    return result


_global_cache = None


def get_global_cache(
    cache_dir: str = ".research_cache",
    ttl_hours: int = 24,
) -> CacheManager:
    global _global_cache

    if _global_cache is None:
        _global_cache = CacheManager(
            cache_dir,
            ttl_hours,
        )

    return _global_cache