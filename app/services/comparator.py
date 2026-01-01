"""Implementation of image comparison copied from project's `compare.py`.

This module implements the same algorithms, caching and download logic as
`compare.py` so the service uses identical behavior while keeping code local
to the service layer.
"""

from __future__ import annotations

from typing import Dict, Optional, Literal, Final
from dataclasses import dataclass
import asyncio
import aiohttp
import hashlib
import io
import time
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from PIL import Image
import numpy as np
import imagehash
import redis
from collections import OrderedDict
import threading
import requests as requests_lib  # type: ignore[import-untyped]

logger = logging.getLogger("comparador.service")

@dataclass(frozen=True, slots=True)
class Config:
    """Configurações otimizadas e imutáveis para thread-safety."""
    default_algorithm: Literal["phash", "dhash"] = "phash"
    hash_size: int = 16

    phash_threshold: float = 0.90
    dhash_threshold: float = 0.88

    use_redis: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
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


config = Config()

# Constantes pré-computadas (evita recriação em cada chamada)
THRESHOLD_MAP: Final[Dict[str, float]] = {
    "phash": config.phash_threshold,
    "dhash": config.dhash_threshold,
}

HASH_FUNCTIONS: Final[Dict[str, str]] = {
    "phash": "calculate_phash",
    "dhash": "calculate_dhash",
}


# ============================================================================
# CACHE MULTI-CAMADA
# ============================================================================


class CacheLayer:
    def __init__(self):
        # Use thread-safe LRU caches for in-memory layer (hashes and comparisons)
        self._memory_cache: "OrderedDict[str, str]" = OrderedDict()
        self._memory_comp_cache: "OrderedDict[str, Dict]" = OrderedDict()
        self._lock = threading.Lock()
        self.redis_client = None
        if config.use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=config.redis_host,
                    port=config.redis_port,
                    db=config.redis_db,
                    decode_responses=False,
                    socket_connect_timeout=2,
                )
                self.redis_client.ping()
                logger.info("✓ Redis conectado")
            except Exception as e:
                logger.warning(f"Redis não disponível: {e}")
                self.redis_client = None

    def _get_cache_key(self, url: str, algorithm: str) -> str:
        # sha1 é mais rápido que md5 e suficiente para cache keys
        url_hash = hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()[:16]
        return f"img_hash:{algorithm}:{url_hash}"

    def get(self, url: str, algorithm: str) -> Optional[str]:
        cache_key = self._get_cache_key(url, algorithm)
        with self._lock:
            if cache_key in self._memory_cache:
                # mark as recently used
                self._memory_cache.move_to_end(cache_key)
                return self._memory_cache[cache_key]
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    hash_value = cached.decode()
                    with self._lock:
                        self._memory_cache[cache_key] = hash_value
                        self._memory_cache.move_to_end(cache_key)
                        if len(self._memory_cache) > config.memory_cache_size:
                            self._memory_cache.popitem(last=False)
                    return hash_value
            except Exception as e:
                logger.error(f"Erro lendo Redis: {e}")
        return None

    def set(self, url: str, algorithm: str, hash_value: str):
        cache_key = self._get_cache_key(url, algorithm)
        with self._lock:
            self._memory_cache[cache_key] = hash_value
            self._memory_cache.move_to_end(cache_key)
            if len(self._memory_cache) > config.memory_cache_size:
                self._memory_cache.popitem(last=False)
        if self.redis_client:
            try:
                self.redis_client.setex(cache_key, config.cache_ttl, hash_value)
            except Exception as e:
                logger.error(f"Erro escrevendo Redis: {e}")

    def get_comparison_result(self, url1: str, url2: str, algorithm: str) -> Optional[Dict]:
        # Ordenar para consistência, usar sha1 (mais rápido)
        urls = "|".join(sorted([url1, url2]))
        cache_key = f"cmp:{algorithm}:{hashlib.sha1(urls.encode(), usedforsecurity=False).hexdigest()[:16]}"
        # Check in-memory comparison cache first
        with self._lock:
            if cache_key in self._memory_comp_cache:
                self._memory_comp_cache.move_to_end(cache_key)
                return self._memory_comp_cache[cache_key]

        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    result = pickle.loads(cached)
                    with self._lock:
                        self._memory_comp_cache[cache_key] = result
                        self._memory_comp_cache.move_to_end(cache_key)
                        if len(self._memory_comp_cache) > config.memory_cache_size:
                            self._memory_comp_cache.popitem(last=False)
                    return result
            except Exception as e:
                logger.error(f"Erro lendo comparação: {e}")

        return None

    def set_comparison_result(self, url1: str, url2: str, algorithm: str, result: Dict):
        urls = "|".join(sorted([url1, url2]))
        cache_key = f"cmp:{algorithm}:{hashlib.sha1(urls.encode(), usedforsecurity=False).hexdigest()[:16]}"
        with self._lock:
            self._memory_comp_cache[cache_key] = result
            self._memory_comp_cache.move_to_end(cache_key)
            if len(self._memory_comp_cache) > config.memory_cache_size:
                self._memory_comp_cache.popitem(last=False)

        if self.redis_client:
            try:
                self.redis_client.setex(cache_key, config.cache_ttl, pickle.dumps(result))
            except Exception as e:
                logger.error(f"Erro salvando comparação: {e}")

    def clear(self):
        with self._lock:
            self._memory_cache.clear()
            self._memory_comp_cache.clear()
        if self.redis_client:
            try:
                # Usar pipeline + UNLINK (non-blocking delete) para performance
                pipe = self.redis_client.pipeline()
                for key in self.redis_client.scan_iter("img_hash:*", count=100):
                    pipe.unlink(key)  # UNLINK é async no Redis, não bloqueia
                for key in self.redis_client.scan_iter("cmp:*", count=100):
                    pipe.unlink(key)
                pipe.execute()
            except Exception as e:
                logger.error(f"Erro limpando Redis: {e}")


# ============================================================================
# DOWNLOAD OTIMIZADO DE IMAGENS
# ============================================================================


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
            response = requests_lib.get(url, timeout=config.download_timeout, stream=True)
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


# ============================================================================
# ALGORITMOS DE HASH
# ============================================================================


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
        """Pré-processa imagem para hash: trim, pad, resize.
        
        Otimizado para minimizar alocações e conversões.
        """
        try:
            img = image
            w, h = img.size
            
            # 1. Trim border (remoção de bordas brancas)
            if config.trim_border:
                img = ImageHasher._trim_border(img, config.border_threshold)
                w, h = img.size
            
            # 2. Pad to square (centraliza em quadrado)
            target_size = config.hash_image_size
            if config.pad_to_square and w != h:
                size = max(w, h)
                # Criar background apenas se necessário
                background = Image.new("RGB", (size, size), (255, 255, 255))
                background.paste(img, ((size - w) >> 1, (size - h) >> 1))  # bit shift é mais rápido
                img = background
                w = h = size
            
            # 3. Resize final (apenas se tamanho diferente)
            if target_size and (w != target_size or h != target_size):
                img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            return img
        except Exception as e:
            logger.warning(f"Preprocess fallback: {e}")
            return image

    @staticmethod
    def _trim_border(image: Image.Image, threshold: int) -> Image.Image:
        """Remove bordas brancas/claras da imagem usando numpy vetorizado.
        
        Performance: ~15x mais rápido que Pillow point() em imagens 1024x1024.
        
        Args:
            image: Imagem PIL em modo RGB.
            threshold: Pixels com valor > threshold são considerados borda (0-255).
        
        Returns:
            Imagem cortada sem bordas, ou original se não houver bordas.
        
        Raises:
            Exceções propagam para o caller (_preprocess_for_hash) que já trata.
        """
        # Converter para grayscale como array numpy (zero-copy quando possível)
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
        row_start = rows_with_content.argmax()
        row_end = len(rows_with_content) - rows_with_content[::-1].argmax()
        col_start = cols_with_content.argmax()
        col_end = len(cols_with_content) - cols_with_content[::-1].argmax()
        
        # Verificar se crop é significativo (evita micro-crops por ruído)
        orig_h, orig_w = gray.shape
        crop_h, crop_w = row_end - row_start, col_end - col_start
        
        if crop_w < orig_w * 0.99 or crop_h < orig_h * 0.99:
            return image.crop((col_start, row_start, col_end, row_end))
        
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
        max_distance = config.hash_size ** 2  # ** é mais legível
        return 1.0 - (distance / max_distance)


# ============================================================================
# COMPARADOR PRINCIPAL
# ============================================================================


class ImageComparator:
    def __init__(self):
        self.cache = CacheLayer()
        self.downloader = ImageDownloader()
        self.hasher = ImageHasher()
        # Executor for CPU-bound hashing operations
        self.executor = ThreadPoolExecutor(max_workers=config.thread_pool_size)
        # Dispatch table (evita getattr + string interpolation)
        self._hash_dispatch: Dict[str, callable] = {
            "phash": self.hasher.calculate_phash,
            "dhash": self.hasher.calculate_dhash,
        }

    async def compare_async(self, url1: str, url2: str, algorithm: str = None, threshold: float = None) -> Dict:
        start_time = time.time()
        algorithm = algorithm or config.default_algorithm

        cached_result = self.cache.get_comparison_result(url1, url2, algorithm)
        if cached_result:
            cached_result["cache_hit"] = True
            cached_result["time"] = time.time() - start_time
            return cached_result

        # Only single-algorithm comparisons are supported (phash, dhash)
        result = await self._compare_single_async(url1, url2, algorithm, threshold)

        result["time"] = time.time() - start_time
        result["cache_hit"] = False

        self.cache.set_comparison_result(url1, url2, algorithm, result)

        return result

    def compare_sync(self, url1: str, url2: str, algorithm: str = None, threshold: float = None) -> Dict:
        if config.enable_async:
            # asyncio.run() é preferido em Python 3.10+, mas pode conflitar com loops existentes
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # Nenhum loop rodando, criar um novo
                return asyncio.run(self.compare_async(url1, url2, algorithm, threshold))
            else:
                # Loop já existe, usar run_until_complete
                return loop.run_until_complete(self.compare_async(url1, url2, algorithm, threshold))
        else:
            return self._compare_sync_internal(url1, url2, algorithm, threshold)

    async def _compare_single_async(self, url1: str, url2: str, algorithm: str, threshold: float = None) -> Dict:
        """Compara duas imagens usando um único algoritmo."""
        if threshold is None:
            threshold = THRESHOLD_MAP.get(algorithm, config.phash_threshold)

        # Baixar/calcular hashes em paralelo (grande ganho de performance)
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

    async def _get_or_calculate_hash_async(self, url: str, algorithm: str) -> Optional[str]:
        cached_hash = self.cache.get(url, algorithm)
        if cached_hash:
            return cached_hash
        image = await self.downloader.download_async(url)
        if image is None:
            return None

        # Dispatch direto (mais rápido que getattr + string interpolation)
        hash_func = self._hash_dispatch[algorithm]
        loop = asyncio.get_running_loop()
        # Offload CPU-bound hashing to executor to avoid blocking the event loop
        hash_value = await loop.run_in_executor(self.executor, hash_func, image)

        if hash_value is not None:
            self.cache.set(url, algorithm, hash_value)
        return hash_value

    def _compare_sync_internal(self, url1: str, url2: str, algorithm: str = None, threshold: float = None) -> Dict:
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


# ============================================================================
# Service wrapper exposing the same API expected by the router
# ============================================================================


class ImageComparatorService:
    """Service wrapper that uses the local ImageComparator implementation."""

    def __init__(self):
        self._impl = ImageComparator()

    async def compare(self, url1: str, url2: str, algorithm: str = None, threshold: float = None) -> Dict:
        return await self._impl.compare_async(url1, url2, algorithm, threshold)

    async def close(self) -> None:
        await self._impl.close()


class HTTPError(Exception):
    pass
