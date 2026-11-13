import redis.asyncio as aioredis
import hashlib
import json
import logging
from typing import Optional, Any, List, Dict

logger = logging.getLogger(__name__)


class VideoCache:
    """Обёртка над Redis для дедупликации обработанных видео"""

    def __init__(
        self,
        host: str,
        port: int,
        ttl_days: int = 7
    ):
        self.client = aioredis.Redis(host=host, port=port, decode_responses=True)
        self.ttl_seconds = ttl_days * 86400

    async def get_cached_chunks(self, video_bytes: bytes) -> Optional[list[dict[str, Any]]]:
        """
        Проверяет, обрабатывалось ли видео ранее
        Возвращает список чанков (словарей) или None
        """
        content_hash = hashlib.sha256(video_bytes).hexdigest()
        key = f"video:{content_hash}"
        try:
            cached = await self.client.get(key)
            if cached:
                logger.info(f"Cache hit for video hash {content_hash[:12]}...")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Redis get failed: {e}")
            return None

    async def set_cached_chunks(self, video_bytes: bytes, chunks: List[Dict[str, Any]]) -> None:
        """Сохраняет результат OCR в Redis"""
        content_hash = hashlib.sha256(video_bytes).hexdigest()
        key = f"video:{content_hash}"
        try:
            await self.client.set(key, json.dumps(chunks), ex=self.ttl_seconds)
            logger.info(f"Cached chunks for video hash {content_hash[:12]}...")
        except Exception as e:
            logger.error(f"Redis set failed: {e}")

    async def close(self):
        await self.client.close()