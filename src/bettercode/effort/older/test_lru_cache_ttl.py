"""
Tests for lru_cache_ttl.py

Run with: python -m pytest test_lru_cache_ttl.py -v
"""

import time
import pytest
from lru_cache_ttl import LRUCacheTTL


class TestBasicOperations:
    def test_put_and_get(self):
        cache = LRUCacheTTL(max_size=4, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2

    def test_get_missing_key(self):
        cache = LRUCacheTTL(max_size=4, default_ttl=10.0)
        assert cache.get("missing") is None
        assert cache.get("missing", "fallback") == "fallback"

    def test_delete(self):
        cache = LRUCacheTTL(max_size=4, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.delete("a") is False
        assert cache.get("a") is None

    def test_update_existing(self):
        cache = LRUCacheTTL(max_size=4, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("a", 99)
        assert cache.get("a") == 99
        assert len(cache) == 1

    def test_contains(self):
        cache = LRUCacheTTL(max_size=4, default_ttl=10.0)
        cache.put("a", 1)
        assert "a" in cache
        assert "b" not in cache

    def test_clear(self):
        cache = LRUCacheTTL(max_size=4, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert len(cache) == 0


class TestTTLExpiry:
    def test_expired_entry_returns_default(self):
        cache = LRUCacheTTL(max_size=4, default_ttl=0.05)
        cache.put("a", 1)
        time.sleep(0.1)
        assert cache.get("a") is None

    def test_non_expired_entry_accessible(self):
        cache = LRUCacheTTL(max_size=4, default_ttl=10.0)
        cache.put("a", 1)
        assert cache.get("a") == 1

    def test_custom_ttl_per_entry(self):
        cache = LRUCacheTTL(max_size=4, default_ttl=10.0)
        cache.put("short", 1, ttl=0.05)
        cache.put("long", 2, ttl=10.0)
        time.sleep(0.1)
        assert cache.get("short") is None
        assert cache.get("long") == 2

    def test_expired_entry_not_in_contains(self):
        cache = LRUCacheTTL(max_size=4, default_ttl=0.05)
        cache.put("a", 1)
        time.sleep(0.1)
        assert "a" not in cache


class TestLRUEviction:
    def test_evicts_lru_when_full(self):
        cache = LRUCacheTTL(max_size=3, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        # cache full, inserting d should evict a (oldest)
        cache.put("d", 4)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("d") == 4

    def test_access_refreshes_lru_order(self):
        cache = LRUCacheTTL(max_size=3, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        # access a, making b the LRU
        cache.get("a")
        cache.put("d", 4)
        assert cache.get("b") is None  # b evicted
        assert cache.get("a") == 1     # a survived


class TestEvictionWithExpiredEntries:
    """These tests verify the critical behavior: when the cache is full and
    expired entries exist, those expired entries should be cleaned up INSTEAD
    of evicting valid live entries."""

    def test_expired_entries_freed_before_live_eviction(self):
        """THE BUG: when expired entries exist and the cache is full,
        inserting a new entry should purge the expired entries and NOT
        evict any live entries."""
        cache = LRUCacheTTL(max_size=4, default_ttl=10.0)

        # fill cache: 2 short-lived, 2 long-lived
        cache.put("expire1", "x", ttl=0.05)
        cache.put("expire2", "x", ttl=0.05)
        cache.put("live1", "important1", ttl=60.0)
        cache.put("live2", "important2", ttl=60.0)

        # wait for the short-lived entries to expire
        time.sleep(0.1)

        # insert a new entry — cache is "full" (4 entries in dict, 2 expired)
        # expected: purge expire1 + expire2, insert new_entry, live1+live2 survive
        cache.put("new_entry", "new_value")

        # THE CRITICAL ASSERTIONS:
        # both live entries must survive — they are not expired
        assert cache.get("live1") == "important1", (
            "live1 was wrongly evicted even though expired entries existed"
        )
        assert cache.get("live2") == "important2", (
            "live2 was wrongly evicted even though expired entries existed"
        )
        assert cache.get("new_entry") == "new_value"

    def test_no_unnecessary_eviction_single_expired(self):
        """Even a single expired entry should prevent live eviction."""
        cache = LRUCacheTTL(max_size=3, default_ttl=10.0)

        cache.put("expire1", "x", ttl=0.05)
        cache.put("live1", "keep_me", ttl=60.0)
        cache.put("live2", "keep_me_too", ttl=60.0)

        time.sleep(0.1)

        cache.put("newcomer", "hello")

        assert cache.get("live1") == "keep_me", (
            "live1 evicted unnecessarily"
        )
        assert cache.get("live2") == "keep_me_too", (
            "live2 evicted unnecessarily"
        )
        assert cache.get("newcomer") == "hello"

    def test_eviction_stats_reflect_expired_cleanup(self):
        """When expired entries are purged to make room, the eviction counter
        for live entries should NOT increment."""
        cache = LRUCacheTTL(max_size=3, default_ttl=10.0)

        cache.put("expire1", "x", ttl=0.05)
        cache.put("expire2", "x", ttl=0.05)
        cache.put("live1", "keep", ttl=60.0)

        time.sleep(0.1)

        cache.put("new1", "val1")

        stats = cache.stats
        assert stats["evictions"] == 0, (
            f"Expected 0 live evictions, got {stats['evictions']}. "
            "Expired entries should have been sufficient."
        )
        assert stats["expired_evictions"] >= 2


class TestEdgeCases:
    def test_all_entries_expired_when_full(self):
        """If all entries are expired, purge them all and insert."""
        cache = LRUCacheTTL(max_size=3, default_ttl=0.05)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        time.sleep(0.1)

        cache.put("d", 4)
        assert cache.get("d") == 4
        assert len(cache) == 1  # only "d" should remain

    def test_mixed_expired_and_live_preserves_order(self):
        """After purging expired entries, LRU order of live entries is preserved."""
        cache = LRUCacheTTL(max_size=4, default_ttl=10.0)

        cache.put("live_oldest", 1, ttl=60.0)
        cache.put("expire1", "x", ttl=0.05)
        cache.put("live_newer", 2, ttl=60.0)
        cache.put("expire2", "x", ttl=0.05)

        time.sleep(0.1)

        # need to insert 2 entries to force LRU eviction of a live entry
        cache.put("new1", "a")
        cache.put("new2", "b")

        # live_oldest should be evicted (it's LRU among live entries)
        # live_newer should survive
        assert cache.get("live_newer") == 2
        assert cache.get("new1") == "a"
        assert cache.get("new2") == "b"

    def test_max_size_one(self):
        cache = LRUCacheTTL(max_size=1, default_ttl=10.0)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_invalid_max_size(self):
        with pytest.raises(ValueError):
            LRUCacheTTL(max_size=0)

    def test_invalid_ttl(self):
        with pytest.raises(ValueError):
            LRUCacheTTL(default_ttl=-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
