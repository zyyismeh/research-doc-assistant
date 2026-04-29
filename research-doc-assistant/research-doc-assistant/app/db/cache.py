"""Redis 缓存管理 - 检索结果缓存 + 会话状态管理"""

from __future__ import annotations

import hashlib
import json

from loguru import logger

from app.core import settings


class CacheManager:
    """Redis 缓存管理器"""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                settings.redis_url, decode_responses=True
            )
        return self._client

    @staticmethod
    def _make_key(prefix: str, query: str) -> str:
        """生成缓存键"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"research:{prefix}:{query_hash}"

    async def get_cached_answer(self, question: str) -> str | None:
        """获取缓存的回答"""
        try:
            key = self._make_key("qa", question)
            cached = await self.client.get(key)
            if cached:
                logger.debug(f"命中缓存: {question[:50]}...")
            return cached
        except Exception as e:
            logger.warning(f"Redis 读取失败: {e}")
            return None

    async def cache_answer(self, question: str, answer: str) -> None:
        """缓存回答"""
        try:
            key = self._make_key("qa", question)
            await self.client.setex(key, settings.redis_cache_ttl, answer)
        except Exception as e:
            logger.warning(f"Redis 写入失败: {e}")

    async def get_session_state(self, session_id: str) -> dict | None:
        """获取会话状态"""
        try:
            key = f"research:session:{session_id}"
            data = await self.client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"会话状态读取失败: {e}")
            return None

    async def set_session_state(self, session_id: str, state: dict) -> None:
        """保存会话状态"""
        try:
            key = f"research:session:{session_id}"
            await self.client.setex(
                key, settings.redis_cache_ttl * 2, json.dumps(state, ensure_ascii=False)
            )
        except Exception as e:
            logger.warning(f"会话状态写入失败: {e}")

    async def close(self):
        if self._client:
            await self._client.close()


cache_manager = CacheManager()
