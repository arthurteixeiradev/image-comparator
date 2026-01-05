from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Final, Literal, Optional

import aiohttp
import imagehash
import numpy as np
import requests as requests_lib
from PIL import Image

from app.core.config import get_settings
from app.services.redis_service import get_redis_service

logger = logging.getLogger("comparador.service")


def _get_config() -> "Config":
    """Carrega configurações do settings centralizado."""
    settings = get_settings()
    return Config(
        use_redis=settings.REDIS_ENABLED,
        cache_ttl=settings.REDIS_CACHE_TTL,
        memory_cache_size=settings.MEMORY_CACHE_SIZE,
    )


@dataclass(frozen=True, slots=True)
class Config:
    """Configurações otimizadas e imutáveis para thread-safety."""

    default_algorithm: Literal["phash", "dhash"] = "dhash"
    hash_size: int = 16

    phash_threshold: float = 0.90
    dhash_threshold: float = 0.88

    use_redis: bool = True
    cache_ttl: int = 86400 * 30
    memory_cache_size: int = 1000

    download_timeout: int = 10
    max_image_size: int = 5 * 1024 * 1024
    resize_for_comparison: bool = True
    max_dimension: int = 1024
    # Preprocessing to handle different sizes and white borders
    trim_border: bool = True
    border_threshold: int = 250
    hash_image_size: int = 256
    pad_to_square: bool = True

    enable_async: bool = True
    thread_pool_size: int = 4


config = _get_config()

# Constantes pré-computadas
THRESHOLD_MAP: Final[Dict[str, float]] = {
    "phash": config.phash_threshold,
    "dhash": config.dhash_threshold,
}

HASH_FUNCTIONS: Final[Dict[str, str]] = {
    "phash": "calculate_phash",
    "dhash": "calculate_dhash",
}

# Cache multi-camada (Memória + Redis)


class CacheLayer:
    """
    Cache multi-camada: Memória (L1) + Redis (L2).

    Estratégia:
    - L1 (Memória): Acesso ultra-rápido, capacidade limitada, LRU eviction.
    - L2 (Redis): Persistente, compartilhado entre instâncias, TTL configurável.

    Fluxo de leitura:
    1. Verifica L1 (memória) → hit = retorna imediatamente
    2. Verifica L2 (Redis) → hit = promove para L1, retorna
    3. Miss = retorna None (caller deve computar o valor)

    Fluxo de escrita:
    1. Escreve em L1 (memória)
    2. Escreve em L2 (Redis) com TTL
    """

    def __init__(self):
        # Cache L1: LRU em memória (thread-safe)
        self._memory_cache: "OrderedDict[str, str]" = OrderedDict()
        self._memory_comp_cache: "OrderedDict[str, Dict]" = OrderedDict()
        self._lock = threading.Lock()

        # Cache L2: Redis centralizado
        self._redis = get_redis_service() if config.use_redis else None

        if self._redis and self._redis.is_connected:
            logger.info("✓ CacheLayer: Redis conectado via RedisService")
        else:
            logger.warning("⚠ CacheLayer: Operando apenas com cache em memória")

    def _get_cache_key(self, url: str, algorithm: str) -> str:
        """Gera chave de cache para hash de imagem."""
        url_hash = hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()[:16]
        return f"img_hash:{algorithm}:{url_hash}"

    def get(self, url: str, algorithm: str) -> Optional[str]:
        """Obtém hash de imagem do cache (L1 → L2)."""
        cache_key = self._get_cache_key(url, algorithm)

        # L1: Verifica cache em memória
        with self._lock:
            if cache_key in self._memory_cache:
                self._memory_cache.move_to_end(cache_key)
                logger.info(f"CACHE HIT [L1/Memória] hash: {cache_key}")
                return self._memory_cache[cache_key]

        # L2: Verifica Redis
        if self._redis and self._redis.is_connected:
            cached = self._redis.get_str(cache_key)
            if cached:
                logger.info(f"CACHE HIT [L2/Redis] hash: {cache_key}")
                # Promove para L1
                with self._lock:
                    self._memory_cache[cache_key] = cached
                    self._memory_cache.move_to_end(cache_key)
                    if len(self._memory_cache) > config.memory_cache_size:
                        self._memory_cache.popitem(last=False)
                return cached

        return None

    def set(self, url: str, algorithm: str, hash_value: str):
        """Armazena hash de imagem no cache (L1 + L2)."""
        cache_key = self._get_cache_key(url, algorithm)

        # L1: Armazena em memória
        with self._lock:
            self._memory_cache[cache_key] = hash_value
            self._memory_cache.move_to_end(cache_key)
            if len(self._memory_cache) > config.memory_cache_size:
                self._memory_cache.popitem(last=False)

        # L2: Armazena no Redis com TTL
        if self._redis and self._redis.is_connected:
            self._redis.set(cache_key, hash_value, ttl=config.cache_ttl)

    def get_comparison_result(
        self, url1: str, url2: str, algorithm: str
    ) -> Optional[Dict]:
        """Obtém resultado de comparação do cache."""
        # Ordenar URLs para consistência
        urls = "|".join(sorted([url1, url2]))
        cache_key = f"cmp:{algorithm}:{hashlib.sha1(urls.encode(), usedforsecurity=False).hexdigest()[:16]}"

        # L1: Verifica cache em memória
        with self._lock:
            if cache_key in self._memory_comp_cache:
                self._memory_comp_cache.move_to_end(cache_key)
                logger.info(f"CACHE HIT [L1/Memória] comparação: {cache_key}")
                return self._memory_comp_cache[cache_key]

        # L2: Verifica Redis
        if self._redis and self._redis.is_connected:
            result = self._redis.get_object(cache_key)
            if result:
                logger.info(f"CACHE HIT [L2/Redis] comparação: {cache_key}")
                # Promove para L1
                with self._lock:
                    self._memory_comp_cache[cache_key] = result
                    self._memory_comp_cache.move_to_end(cache_key)
                    if len(self._memory_comp_cache) > config.memory_cache_size:
                        self._memory_comp_cache.popitem(last=False)
                return result

        return None

    def set_comparison_result(self, url1: str, url2: str, algorithm: str, result: Dict):
        """Armazena resultado de comparação no cache."""
        urls = "|".join(sorted([url1, url2]))
        cache_key = f"cmp:{algorithm}:{hashlib.sha1(urls.encode(), usedforsecurity=False).hexdigest()[:16]}"

        # L1: Armazena em memória
        with self._lock:
            self._memory_comp_cache[cache_key] = result
            self._memory_comp_cache.move_to_end(cache_key)
            if len(self._memory_comp_cache) > config.memory_cache_size:
                self._memory_comp_cache.popitem(last=False)

        # L2: Armazena no Redis com TTL
        if self._redis and self._redis.is_connected:
            self._redis.set_object(cache_key, result, ttl=config.cache_ttl)

    def clear(self):
        """Limpa todo o cache (L1 + L2)."""
        # L1: Limpa memória
        with self._lock:
            self._memory_cache.clear()
            self._memory_comp_cache.clear()

        # L2: Limpa Redis (apenas chaves deste serviço)
        if self._redis and self._redis.is_connected:
            self._redis.delete_pattern("img_hash:*")
            self._redis.delete_pattern("cmp:*")
            logger.info("Cache limpo (memória + Redis)")


# Download otimizado de imagens


class ImageDownloader:
    def __init__(self):
        self.session = None
        self.thread_pool = ThreadPoolExecutor(max_workers=config.thread_pool_size)

    async def download_async(self, url: str) -> Optional[Image.Image]:
        try:
            if self.session is None:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=config.download_timeout),
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"HTTP {response.status} para {url}")
                    return None
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > config.max_image_size:
                    logger.error(f"Imagem muito grande: {content_length} bytes")
                    return None
                image_data = await response.read()
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                if config.resize_for_comparison:
                    image = self._resize_image(image)
                return image
        except asyncio.TimeoutError:
            logger.error(f"Timeout downloading {url}")
            return None
        except Exception as e:
            logger.error(f"Erro baixando {url}: {e}")
            return None

    def download_sync(self, url: str) -> Optional[Image.Image]:
        try:
            response = requests_lib.get(
                url, timeout=config.download_timeout, stream=True
            )
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > config.max_image_size:
                logger.error(f"Imagem muito grande: {content_length} bytes")
                return None
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            if config.resize_for_comparison:
                image = self._resize_image(image)
            return image
        except Exception as e:
            logger.error(f"Erro baixando {url}: {e}")
            return None

    def _resize_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width > config.max_dimension or height > config.max_dimension:
            ratio = min(config.max_dimension / width, config.max_dimension / height)
            new_size = (int(width * ratio), int(height * ratio))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    async def close(self):
        if self.session:
            try:
                await self.session.close()
            except Exception:
                pass
            finally:
                self.session = None
        try:
            self.thread_pool.shutdown()
        except Exception:
            pass


# Algoritmos de hash


class ImageHasher:
    @staticmethod
    def calculate_phash(image: Image.Image) -> str:
        img = ImageHasher._preprocess_for_hash(image)
        return str(imagehash.phash(img, hash_size=config.hash_size))

    @staticmethod
    def calculate_dhash(image: Image.Image) -> str:
        img = ImageHasher._preprocess_for_hash(image)
        return str(imagehash.dhash(img, hash_size=config.hash_size))

    @staticmethod
    def _preprocess_for_hash(image: Image.Image) -> Image.Image:
        """Pré-processa imagem para hash: trim, pad, resize."""
        try:
            img = image
            w, h = img.size

            # Trim border
            if config.trim_border:
                img = ImageHasher._trim_border(img, config.border_threshold)
                w, h = img.size

            # Pad to square (centraliza em quadrado)
            target_size = config.hash_image_size
            if config.pad_to_square and w != h:
                size = max(w, h)
                # Criar background apenas se necessário
                background = Image.new("RGB", (size, size), (255, 255, 255))
                background.paste(
                    img, ((size - w) >> 1, (size - h) >> 1)
                )  # bit shift é mais rápido
                img = background
                w = h = size

            # Resize final (apenas se tamanho diferente)
            if target_size and (w != target_size or h != target_size):
                img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

            return img
        except Exception as e:
            logger.warning(f"Preprocess fallback: {e}")
            return image

    @staticmethod
    def _trim_border(image: Image.Image, threshold: int) -> Image.Image:
        """Remove bordas brancas/claras da imagem usando numpy vetorizado.

        Args:
            image: Imagem PIL em modo RGB.
            threshold: Pixels com valor > threshold são considerados borda (0-255).

        Returns:
            Imagem cortada sem bordas, ou original se não houver bordas.
        """
        # Converter para grayscale como array numpy
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        # Máscara vetorizada: True onde há conteúdo (não-branco)
        content_mask = gray <= threshold

        # Imagem toda branca - retorna original
        if not content_mask.any():
            return image

        # Encontra linhas/colunas com conteúdo (operações vetorizadas SIMD)
        rows_with_content = content_mask.any(axis=1)
        cols_with_content = content_mask.any(axis=0)

        # Bounding box: primeiro e último índice True
        rows_arr = np.asarray(rows_with_content)
        cols_arr = np.asarray(cols_with_content)
        row_start = int(rows_arr.argmax())
        row_end = len(rows_arr) - int(rows_arr[::-1].argmax())
        col_start = int(cols_arr.argmax())
        col_end = len(cols_arr) - int(cols_arr[::-1].argmax())

        # Verificar se crop é significativo
        orig_h, orig_w = gray.shape
        crop_h, crop_w = row_end - row_start, col_end - col_start

        if crop_w < orig_w * 0.99 or crop_h < orig_h * 0.99:
            return image.crop(
                (float(col_start), float(row_start), float(col_end), float(row_end))
            )

        return image

    @staticmethod
    @lru_cache(maxsize=1024)
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Calcula distância de Hamming entre dois hashes (cached)."""
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2

    @staticmethod
    def calculate_similarity(hash1: str, hash2: str) -> float:
        """Converte distância Hamming para similaridade [0, 1]."""
        distance = ImageHasher.hamming_distance(hash1, hash2)
        max_distance = config.hash_size**2  # ** é mais legível
        return 1.0 - (distance / max_distance)


# Comparador principal


class ImageComparator:
    def __init__(self):
        self.cache = CacheLayer()
        self.downloader = ImageDownloader()
        self.hasher = ImageHasher()
        # Executor for CPU-bound hashing operations
        self.executor = ThreadPoolExecutor(max_workers=config.thread_pool_size)
        # Dispatch table
        self._hash_dispatch: Dict[str, Any] = {
            "phash": self.hasher.calculate_phash,
            "dhash": self.hasher.calculate_dhash,
        }

    async def compare_async(
        self,
        url1: str,
        url2: str,
        algorithm: str | None = None,
        threshold: float | None = None,
    ) -> Dict:
        start_time = time.time()
        algorithm = algorithm or config.default_algorithm

        cached_result = self.cache.get_comparison_result(url1, url2, algorithm)
        if cached_result:
            cached_result["cache_hit"] = True
            cached_result["time"] = time.time() - start_time
            return cached_result

        # Only single-algorithm comparisons are supported
        result = await self._compare_single_async(url1, url2, algorithm, threshold)

        result["time"] = time.time() - start_time
        result["cache_hit"] = False

        self.cache.set_comparison_result(url1, url2, algorithm, result)

        return result

    def compare_sync(
        self,
        url1: str,
        url2: str,
        algorithm: str | None = None,
        threshold: float | None = None,
    ) -> Dict:
        if config.enable_async:
            # asyncio.run() é preferido em Python 3.10+, mas pode conflitar com loops existentes
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # Nenhum loop rodando, criar um novo
                return asyncio.run(self.compare_async(url1, url2, algorithm, threshold))
            else:
                # Loop já existe, usar run_until_complete
                return loop.run_until_complete(
                    self.compare_async(url1, url2, algorithm, threshold)
                )
        else:
            return self._compare_sync_internal(url1, url2, algorithm, threshold)

    async def _compare_single_async(
        self, url1: str, url2: str, algorithm: str, threshold: float | None = None
    ) -> Dict:
        """Compara duas imagens usando um único algoritmo."""
        if threshold is None:
            threshold = THRESHOLD_MAP.get(algorithm, config.phash_threshold)

        # Baixar/calcular hashes em paralelo
        hash1, hash2 = await asyncio.gather(
            self._get_or_calculate_hash_async(url1, algorithm),
            self._get_or_calculate_hash_async(url2, algorithm),
        )

        if hash1 is None or hash2 is None:
            return {
                "are_same": False,
                "similarity": 0.0,
                "distance": config.hash_size * config.hash_size,
                "algorithm": algorithm,
                "error": "Failed to download or process images",
            }

        distance = self.hasher.hamming_distance(hash1, hash2)
        similarity = self.hasher.calculate_similarity(hash1, hash2)

        return {
            "are_same": bool(similarity >= threshold),
            "similarity": round(similarity, 4),
            "distance": int(distance),
            "algorithm": algorithm,
            "threshold": threshold,
        }

    async def _get_or_calculate_hash_async(
        self, url: str, algorithm: str
    ) -> Optional[str]:
        cached_hash = self.cache.get(url, algorithm)
        if cached_hash:
            return cached_hash
        image = await self.downloader.download_async(url)
        if image is None:
            return None

        # Dispatch direto
        hash_func = self._hash_dispatch[algorithm]
        loop = asyncio.get_running_loop()
        # Offload CPU-bound hashing to executor to avoid blocking the event loop
        hash_value = await loop.run_in_executor(self.executor, hash_func, image)

        if hash_value is not None:
            self.cache.set(url, algorithm, hash_value)
        return hash_value

    def _compare_sync_internal(
        self,
        url1: str,
        url2: str,
        algorithm: str | None = None,
        threshold: float | None = None,
    ) -> Dict:
        start_time = time.time()
        algorithm = algorithm or config.default_algorithm

        cached_result = self.cache.get_comparison_result(url1, url2, algorithm)
        if cached_result:
            cached_result["cache_hit"] = True
            return cached_result

        img1 = self.downloader.download_sync(url1)
        img2 = self.downloader.download_sync(url2)

        if img1 is None or img2 is None:
            return {
                "are_same": False,
                "similarity": 0.0,
                "distance": config.hash_size * config.hash_size,
                "algorithm": algorithm,
                "error": "Failed to download images",
                "time": time.time() - start_time,
            }

        hash_func = self._hash_dispatch[algorithm]
        h1 = hash_func(img1)
        h2 = hash_func(img2)
        similarity = self.hasher.calculate_similarity(h1, h2)
        threshold = threshold or THRESHOLD_MAP.get(algorithm, config.phash_threshold)

        distance = int((1 - similarity) * config.hash_size * config.hash_size)

        result = {
            "are_same": bool(similarity >= threshold),
            "similarity": round(similarity, 4),
            "distance": distance,
            "algorithm": algorithm,
            "threshold": threshold,
            "time": time.time() - start_time,
            "cache_hit": False,
        }

        self.cache.set_comparison_result(url1, url2, algorithm, result)
        return result

    async def close(self) -> None:
        """Libera recursos de forma segura."""
        await self.downloader.close()
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python < 3.9 não suporta cancel_futures
            self.executor.shutdown(wait=False)
        except Exception:
            pass


# Service wrapper exposing the same API expected by the router


class ImageComparatorService:
    """Service wrapper that uses the local ImageComparator implementation."""

    def __init__(self):
        self._impl = ImageComparator()

    async def compare(
        self,
        url1: str,
        url2: str,
        algorithm: str | None = None,
        threshold: float | None = None,
    ) -> Dict:
        return await self._impl.compare_async(url1, url2, algorithm, threshold)

    async def close(self) -> None:
        await self._impl.close()


class HTTPError(Exception):
    pass
