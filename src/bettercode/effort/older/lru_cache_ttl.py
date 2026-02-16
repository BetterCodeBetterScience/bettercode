"""
lru_cache_ttl.py - LRU Cache with per-entry TTL (time-to-live) expiry.

This module implements a thread-safe LRU cache where each entry can have
an individual TTL. When an entry expires, it should be lazily evicted on
the next access. The cache also supports a max size, evicting the least
recently used *non-expired* entry when full.

Known issue (reported by user):
    "When the cache is full and I insert a new key, sometimes valid
    (non-expired) entries get evicted even though there are expired entries
    that should have been cleaned up first. This seems to happen specifically
    when expired entries exist but the cache still evicts a live entry instead."

Your task: find and fix the bug described above. The fix should be minimal —
do not refactor or reorganize the code. Just fix the broken logic.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Optional, Hashable


class TTLEntry:
    """Wraps a cached value with its expiration timestamp."""

    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: float, now: Optional[float] = None):
        self.value = value
        now = now or time.monotonic()
        self.expires_at = now + ttl

    def is_expired(self, now: Optional[float] = None) -> bool:
        now = now or time.monotonic()
        return now >= self.expires_at


class LRUCacheTTL:
    """Thread-safe LRU cache with per-entry TTL support.

    Parameters
    ----------
    max_size : int
        Maximum number of entries the cache can hold.
    default_ttl : float
        Default time-to-live in seconds for entries that don't specify one.
    """

    def __init__(self, max_size: int = 128, default_ttl: float = 300.0):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")

        self._max_size = max_size
        self._default_ttl = default_ttl
        self._data: OrderedDict[Hashable, TTLEntry] = OrderedDict()
        self._lock = threading.Lock()

        # stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired_evictions = 0

    # -- public API --

    def get(self, key: Hashable, default: Any = None) -> Any:
        """Retrieve a value by key. Returns *default* if missing or expired."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return default

            if entry.is_expired():
                # lazy expiry
                del self._data[key]
                self._expired_evictions += 1
                self._misses += 1
                return default

            # mark as recently used
            self._data.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: Hashable, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a cache entry.

        If the cache is full, expired entries are purged first. If still full
        after purging, the least recently used live entry is evicted.
        """
        ttl = ttl if ttl is not None else self._default_ttl
        now = time.monotonic()

        with self._lock:
            # update existing key
            if key in self._data:
                self._data[key] = TTLEntry(value, ttl, now=now)
                self._data.move_to_end(key)
                return

            # need to make room?
            if len(self._data) >= self._max_size:
                self._evict(now)

            self._data[key] = TTLEntry(value, ttl, now=now)

    def delete(self, key: Hashable) -> bool:
        """Remove an entry. Returns True if the key existed."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._data[key]
                self._expired_evictions += 1
                return False
            return True

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._data),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "expired_evictions": self._expired_evictions,
            }

    # -- internal --

    def _evict(self, now: float) -> None:
        """Make room for one new entry.

        Strategy: first, purge all expired entries. If that freed space, done.
        Otherwise evict the single least-recently-used live entry.
        """
        # phase 1: purge expired
        expired_keys = [
            k for k, entry in self._data.items() if entry.is_expired(now)
        ]
        for k in expired_keys:
            del self._data[k]
            self._expired_evictions += 1

        # phase 2: evict LRU live entry to make room
        if self._data:
            oldest_key, _ = next(iter(self._data.items()))
            del self._data[oldest_key]
            self._evictions += 1

    def _purge_expired(self, now: Optional[float] = None) -> int:
        """Remove all expired entries. Returns count removed."""
        now = now or time.monotonic()
        expired_keys = [
            k for k, entry in self._data.items() if entry.is_expired(now)
        ]
        for k in expired_keys:
            del self._data[k]
            self._expired_evictions += 1
        return len(expired_keys)

    def keys(self) -> list:
        """Return a list of all non-expired keys."""
        now = time.monotonic()
        with self._lock:
            return [
                k for k, entry in self._data.items()
                if not entry.is_expired(now)
            ]

    def values(self) -> list:
        """Return a list of all non-expired values."""
        now = time.monotonic()
        with self._lock:
            return [
                entry.value for entry in self._data.values()
                if not entry.is_expired(now)
            ]

    def items(self) -> list:
        """Return a list of (key, value) for all non-expired entries."""
        now = time.monotonic()
        with self._lock:
            return [
                (k, entry.value) for k, entry in self._data.items()
                if not entry.is_expired(now)
            ]
