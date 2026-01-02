from __future__ import annotations

import logging
import pickle
from typing import Any, Optional, Union

import redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from app.core.config import get_settings

logger = logging.getLogger("comparador.redis")


class RedisService:
    _instance: Optional["RedisService"] = None
    _client: Optional[redis.Redis] = None
    _is_connected: bool = False
    
    def __new__(cls) -> "RedisService":
        """Implementa Singleton pattern para garantir conexão única."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Inicializa a conexão Redis com as configurações do ambiente."""
        settings = get_settings()
        
        if not settings.REDIS_ENABLED:
            logger.info("Redis desabilitado via configuração")
            self._is_connected = False
            return
        
        try:
            if settings.REDIS_URL:
                self._client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=False,
                    socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT,
                    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
            else:
                self._client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD or None,
                    decode_responses=False,
                    socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT,
                    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
            
            self._client.ping()
            self._is_connected = True
            logger.info("✓ Redis conectado com sucesso")
            
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"⚠ Redis não disponível: {e}")
            self._client = None
            self._is_connected = False
        except Exception as e:
            logger.error(f"✗ Erro ao conectar Redis: {e}")
            self._client = None
            self._is_connected = False
    
    @property
    def is_connected(self) -> bool:
        """Retorna True se há conexão ativa com Redis."""
        return self._is_connected and self._client is not None
    
    @property
    def client(self) -> Optional[redis.Redis]:
        """Retorna o cliente Redis raw para operações avançadas."""
        return self._client
    
    def get(self, key: str) -> Optional[bytes]:
        """
        Obtém um valor do Redis.
        
        Args:
            key: Chave para buscar.
        
        Returns:
            Valor em bytes ou None se não encontrado/erro.
        """
        if not self.is_connected:
            return None
        
        try:
            return self._client.get(key)
        except RedisError as e:
            logger.error(f"Erro ao ler chave '{key}': {e}")
            return None
    
    def get_str(self, key: str) -> Optional[str]:
        """
        Obtém um valor do Redis como string.
        
        Args:
            key: Chave para buscar.
        
        Returns:
            Valor como string ou None se não encontrado/erro.
        """
        value = self.get(key)
        if value is not None:
            return value.decode("utf-8")
        return None
    
    def get_object(self, key: str) -> Optional[Any]:
        """
        Obtém um objeto serializado do Redis (pickle).
        
        Args:
            key: Chave para buscar.
        
        Returns:
            Objeto deserializado ou None se não encontrado/erro.
        """
        value = self.get(key)
        if value is not None:
            try:
                return pickle.loads(value)
            except Exception as e:
                logger.error(f"Erro ao deserializar objeto '{key}': {e}")
        return None
    
    def set(
        self,
        key: str,
        value: Union[str, bytes],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Define um valor no Redis.
        
        Args:
            key: Chave para armazenar.
            value: Valor (string ou bytes).
            ttl: Tempo de expiração em segundos (opcional).
        
        Returns:
            True se sucesso, False caso contrário.
        """
        if not self.is_connected:
            return False
        
        try:
            if ttl:
                self._client.setex(key, ttl, value)
            else:
                self._client.set(key, value)
            return True
        except RedisError as e:
            logger.error(f"Erro ao definir chave '{key}': {e}")
            return False
    
    def set_object(
        self,
        key: str,
        obj: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Serializa e armazena um objeto no Redis (pickle).
        
        Args:
            key: Chave para armazenar.
            obj: Objeto Python para serializar.
            ttl: Tempo de expiração em segundos (opcional).
        
        Returns:
            True se sucesso, False caso contrário.
        """
        try:
            serialized = pickle.dumps(obj)
            return self.set(key, serialized, ttl)
        except Exception as e:
            logger.error(f"Erro ao serializar objeto '{key}': {e}")
            return False
    
    def delete(self, *keys: str) -> int:
        """
        Remove uma ou mais chaves do Redis.
        
        Args:
            keys: Uma ou mais chaves para remover.
        
        Returns:
            Número de chaves removidas.
        """
        if not self.is_connected or not keys:
            return 0
        
        try:
            return self._client.delete(*keys)
        except RedisError as e:
            logger.error(f"Erro ao deletar chaves {keys}: {e}")
            return 0
    
    def exists(self, *keys: str) -> int:
        """
        Verifica se uma ou mais chaves existem no Redis.
        
        Args:
            keys: Uma ou mais chaves para verificar.
        
        Returns:
            Número de chaves que existem.
        """
        if not self.is_connected or not keys:
            return 0
        
        try:
            return self._client.exists(*keys)
        except RedisError as e:
            logger.error(f"Erro ao verificar chaves {keys}: {e}")
            return 0
    
    def expire(self, key: str, ttl: int) -> bool:
        """
        Define ou atualiza o TTL de uma chave existente.
        
        Args:
            key: Chave para atualizar.
            ttl: Novo TTL em segundos.
        
        Returns:
            True se a chave existe e TTL foi definido, False caso contrário.
        """
        if not self.is_connected:
            return False
        
        try:
            return bool(self._client.expire(key, ttl))
        except RedisError as e:
            logger.error(f"Erro ao definir TTL para '{key}': {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """
        Retorna o TTL restante de uma chave.
        
        Args:
            key: Chave para verificar.
        
        Returns:
            TTL em segundos, -1 se não tem TTL, -2 se não existe.
        """
        if not self.is_connected:
            return -2
        
        try:
            return self._client.ttl(key)
        except RedisError as e:
            logger.error(f"Erro ao obter TTL de '{key}': {e}")
            return -2
    
    def scan_keys(self, pattern: str, count: int = 100) -> list[str]:
        """
        Busca chaves que correspondem a um padrão.
        
        Args:
            pattern: Padrão glob (ex: "img_hash:*").
            count: Número aproximado de chaves por iteração.
        
        Returns:
            Lista de chaves encontradas.
        """
        if not self.is_connected:
            return []
        
        try:
            keys = []
            for key in self._client.scan_iter(pattern, count=count):
                keys.append(key.decode("utf-8") if isinstance(key, bytes) else key)
            return keys
        except RedisError as e:
            logger.error(f"Erro ao buscar chaves '{pattern}': {e}")
            return []
    
    def delete_pattern(self, pattern: str) -> int:
        """
        Remove todas as chaves que correspondem a um padrão.
        
        Args:
            pattern: Padrão glob (ex: "cache:*").
        
        Returns:
            Número de chaves removidas.
        """
        if not self.is_connected:
            return 0
        
        try:
            keys = self.scan_keys(pattern)
            if keys:
                return self._client.unlink(*keys)
            return 0
        except RedisError as e:
            logger.error(f"Erro ao deletar padrão '{pattern}': {e}")
            return 0
    
    def ping(self) -> bool:
        """
        Verifica se a conexão Redis está ativa.
        
        Returns:
            True se conectado e respondendo, False caso contrário.
        """
        if not self._client:
            return False
        
        try:
            return self._client.ping()
        except RedisError:
            self._is_connected = False
            return False
    
    def disconnect(self) -> None:
        """
        Encerra a conexão Redis de forma segura.
        
        Deve ser chamado durante o shutdown da aplicação.
        """
        if self._client:
            try:
                self._client.close()
                logger.info("Redis desconectado")
            except Exception as e:
                logger.error(f"Erro ao desconectar Redis: {e}")
            finally:
                self._client = None
                self._is_connected = False
    
    def reconnect(self) -> bool:
        """
        Tenta reconectar ao Redis.
        
        Útil após falhas de conexão ou timeout.
        
        Returns:
            True se reconectou com sucesso, False caso contrário.
        """
        self.disconnect()
        self._initialize()
        return self.is_connected
    
    def health_check(self) -> dict:
        """
        Retorna informações sobre o estado da conexão.
        
        Returns:
            Dict com status, latência e info do servidor.
        """
        import time
        from app.core.config import get_settings
        
        settings = get_settings()
        
        if not self.is_connected:
            return {
                "status": "disconnected",
                "latency_ms": None,
                "server_info": None,
                "cache_ttl_seconds": settings.REDIS_CACHE_TTL,
            }
        
        try:
            start = time.perf_counter()
            self._client.ping()
            latency = (time.perf_counter() - start) * 1000
            
            info = self._client.info("server")
            
            return {
                "status": "connected",
                "latency_ms": round(latency, 2),
                "cache_ttl_seconds": settings.REDIS_CACHE_TTL,
                "server_info": {
                    "redis_version": info.get("redis_version"),
                    "uptime_days": info.get("uptime_in_days"),
                },
            }
        except RedisError as e:
            return {
                "status": "error",
                "error": str(e),
                "latency_ms": None,
                "server_info": None,
            }


# Instância global para fácil acesso
_redis_service: Optional[RedisService] = None


def get_redis_service() -> RedisService:
    """
    Factory function para obter a instância do RedisService.
    
    Uso recomendado para injeção de dependência com FastAPI.
    
    Returns:
        Instância singleton do RedisService.
    """
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service
