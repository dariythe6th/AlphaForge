# src/memory/short_term.py
import redis
import json
from typing import Dict, Any, Optional
import asyncio
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShortTermMemory:
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            self.ttl = ttl
            logger.info("Short-term memory (Redis) initialized")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            self.local_cache = {}

    async def set(self, key: str, value: Dict[str, Any]) -> bool:
        """Store a value in short-term memory"""
        try:
            if self.redis_client:
                self.redis_client.setex(
                    key,
                    self.ttl,
                    json.dumps(value)
                )
            else:
                self.local_cache[key] = {
                    'value': value,
                    'expires': datetime.now() + timedelta(seconds=self.ttl)
                }
            return True
        except Exception as e:
            logger.error(f"Error setting memory key {key}: {e}")
            return False

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a value from short-term memory"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            elif key in self.local_cache:
                cache_item = self.local_cache[key]
                if cache_item['expires'] > datetime.now():
                    return cache_item['value']
                else:
                    del self.local_cache[key]
            return None
        except Exception as e:
            logger.error(f"Error getting memory key {key}: {e}")
            return None

    async def delete(self, key: str) -> bool:
        """Delete a value from short-term memory"""
        try:
            if self.redis_client:
                return self.redis_client.delete(key) > 0
            elif key in self.local_cache:
                del self.local_cache[key]
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting memory key {key}: {e}")
            return False

    async def clear_session(self, session_id: str) -> bool:
        """Clear all session-related memory"""
        try:
            if self.redis_client:
                # Find all keys for this session
                keys = self.redis_client.keys(f"session:{session_id}:*")
                if keys:
                    self.redis_client.delete(*keys)
                return True
            else:
                # Clear from local cache
                keys_to_delete = [k for k in self.local_cache.keys() if k.startswith(f"session:{session_id}:")]
                for key in keys_to_delete:
                    del self.local_cache[key]
                return True
        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {e}")
            return False