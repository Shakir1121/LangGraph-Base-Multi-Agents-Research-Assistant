"""
Cache manager for storing and retrieving research results
Implements file-based caching with TTL support
"""

import json
import logging
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
import os

logger = logging.getLogger(__name__)


class CacheManager:
    """File-based cache with TTL support for research queries and results"""
    
    def __init__(self, cache_dir: str = ".research_cache", ttl_hours: int = 24):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory for cache files
            ttl_hours: Time-to-live in hours for cached items
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def _is_expired(self, timestamp: str, ttl_seconds: Optional[int] = None) -> bool:
        """Check if cached item is expired"""
        try:
            cached_time = datetime.fromisoformat(timestamp)
            # Use per-item TTL if provided, otherwise use instance TTL
            if ttl_seconds is not None:
                item_ttl = timedelta(seconds=ttl_seconds)
            else:
                item_ttl = self.ttl
            return datetime.now() - cached_time > item_ttl
        except Exception:
            return True
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result
        
        Args:
            query: Research query
        
        Returns:
            Cached result or None if not found/expired
        """
        cache_key = self._get_cache_key(query)
        
        if cache_key not in self.metadata:
            return None
        
        metadata = self.metadata[cache_key]
        
        # Check if expired (pass per-item TTL if available)
        ttl_seconds = metadata.get("ttl_seconds")
        if self._is_expired(metadata["timestamp"], ttl_seconds):
            self.delete(query)
            logger.info(f"🗑️ Cache expired for: {query[:50]}...")
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            logger.warning(f"Cache file missing for key: {cache_key}")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
            logger.info(f"✅ Cache HIT for: {query[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Failed to retrieve cache: {e}")
            return None
    
    def set(self, query: str, result: Dict[str, Any], ttl: Optional[int] = None):
        """
        Store result in cache
        
        Args:
            query: Research query
            result: Result to cache
            ttl: Optional time-to-live in seconds (overrides instance TTL)
        """
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            self.metadata[cache_key] = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "size": cache_file.stat().st_size,
                "ttl_seconds": ttl  # Store per-item TTL if provided
            }
            self._save_metadata()
            logger.info(f"💾 Cached: {query[:50]}... ({cache_file.stat().st_size / 1024:.1f}KB)")
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
    
    def delete(self, query: str) -> bool:
        """
        Delete cached result
        
        Args:
            query: Research query
        
        Returns:
            True if deleted, False otherwise
        """
        cache_key = self._get_cache_key(query)
        
        if cache_key not in self.metadata:
            return False
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            if cache_file.exists():
                cache_file.unlink()
            del self.metadata[cache_key]
            self._save_metadata()
            logger.info(f"🗑️ Deleted cache: {query[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to delete cache: {e}")
            return False
    
    def clear_expired(self):
        """Clear all expired cache entries"""
        expired_keys = []
        
        for cache_key, metadata in self.metadata.items():
            ttl_seconds = metadata.get("ttl_seconds")
            if self._is_expired(metadata["timestamp"], ttl_seconds):
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                if cache_file.exists():
                    cache_file.unlink()
                del self.metadata[cache_key]
            except Exception as e:
                logger.warning(f"Failed to delete expired cache {cache_key}: {e}")
        
        if expired_keys:
            self._save_metadata()
            logger.info(f"🗑️ Cleared {len(expired_keys)} expired cache entries")
    
    def clear_all(self):
        """Clear all cache"""
        try:
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            self.metadata = {}
            self._save_metadata()
            logger.info("🗑️ Cleared all cache")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = sum(
            f.stat().st_size 
            for f in self.cache_dir.glob("*.pkl") 
            if f.is_file()
        )
        
        return {
            "cached_queries": len(self.metadata),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
            "ttl_hours": self.ttl.total_seconds() / 3600
        }


def cached_research_search(query: str, search_func, cache_dir: str = ".research_cache", ttl_hours: int = 24):
    """
    Decorator for caching research search results
    
    Args:
        query: Search query
        search_func: Function to call if not cached
        cache_dir: Cache directory
        ttl_hours: Time-to-live in hours
    
    Returns:
        Cached or fresh result
    """
    cache = CacheManager(cache_dir, ttl_hours)
    
    # Try to get from cache
    cached_result = cache.get(query)
    if cached_result is not None:
        return cached_result
    
    # Get fresh result
    result = search_func()
    
    # Cache it
    cache.set(query, result)
    
    return result


# Global cache instance
_global_cache = None


def get_global_cache(cache_dir: str = ".research_cache", ttl_hours: int = 24) -> CacheManager:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager(cache_dir, ttl_hours)
    return _global_cache
